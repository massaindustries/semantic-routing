package proxy

import (
	"bytes"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// anthropicHTTPClient is the upstream client used to forward Anthropic-native
// /v1/messages requests. A long timeout (15 min) accommodates streaming
// responses that span the full max_tokens window.
var anthropicHTTPClient = &http.Client{
	Timeout: 15 * time.Minute,
	Transport: &http.Transport{
		TLSClientConfig:     &tls.Config{MinVersion: tls.VersionTLS12},
		MaxIdleConns:        50,
		MaxIdleConnsPerHost: 10,
		IdleConnTimeout:     90 * time.Second,
		DisableCompression:  true, // SSE does not benefit from gzip and breaks Flush semantics
	},
}

// hopByHopHeaders MUST NOT be forwarded across a proxy hop (RFC 7230 §6.1).
var hopByHopHeaders = map[string]struct{}{
	"Connection":          {},
	"Proxy-Connection":    {},
	"Keep-Alive":          {},
	"Proxy-Authenticate":  {},
	"Proxy-Authorization": {},
	"Te":                  {},
	"Trailer":             {},
	"Transfer-Encoding":   {},
	"Upgrade":             {},
	"Host":                {},
	"Content-Length":      {},
	"Accept-Encoding":     {},
}

// handleAnthropicMessages implements a transparent pass-through for the
// Anthropic /v1/messages endpoint. The request body and headers (including
// Authorization, anthropic-version, anthropic-beta, User-Agent) are forwarded
// verbatim to the configured upstream — only the `model` field is rewritten
// based on the difficulty classification of the prompt.
func (s *Server) handleAnthropicMessages(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	cfg := s.router.Config
	if cfg == nil {
		writeError(w, http.StatusInternalServerError, "router config not loaded")
		return
	}
	apCfg := &cfg.AnthropicPassthrough
	if !apCfg.Enabled {
		writeError(w, http.StatusNotFound, "anthropic_passthrough is disabled")
		return
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, maxRequestBodySize))
	if err != nil {
		writeError(w, http.StatusBadRequest, fmt.Sprintf("reading request body: %v", err))
		return
	}
	defer r.Body.Close()

	if len(body) == 0 {
		writeError(w, http.StatusBadRequest, "empty request body")
		return
	}
	if len(body) >= maxRequestBodySize {
		writeError(w, http.StatusRequestEntityTooLarge,
			fmt.Sprintf("request body too large (max %d bytes)", maxRequestBodySize))
		return
	}

	prompt := extractAnthropicPromptText(body)
	if prompt == "" {
		logging.Warnf("AnthropicPassthrough: could not extract prompt text, falling back to medium")
	}

	labels := s.router.Classifier.ClassifyComplexity(prompt)
	label := "medium"
	if len(labels) > 0 {
		label = strings.TrimPrefix(labels[0], "complexity:")
	}

	clientWants1M := requestRequestsContext1M(r.Header.Values("Anthropic-Beta"))
	use1M := clientWants1M && apCfg.ExtraUsageEnabled && len(body) > apCfg.EffectiveContext1MThresholdBytes()

	var selectedModel string
	if use1M {
		selectedModel = apCfg.Resolve1M(label)
	} else {
		selectedModel = apCfg.Resolve(label)
	}

	metrics.BrickCCRequests.WithLabelValues(label, selectedModel).Inc()

	rewritten := rewriteModelInBody(body, selectedModel)

	upstreamURL := apCfg.EffectiveUpstreamURL() + "/v1/messages"
	upstreamReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, upstreamURL, bytes.NewReader(rewritten))
	if err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("building upstream request: %v", err))
		return
	}
	for name, values := range r.Header {
		if _, hop := hopByHopHeaders[http.CanonicalHeaderKey(name)]; hop {
			continue
		}
		if http.CanonicalHeaderKey(name) == "Anthropic-Beta" && !use1M {
			for _, v := range values {
				stripped := stripContext1MBeta(v)
				if stripped != "" {
					upstreamReq.Header.Add(name, stripped)
				}
			}
			continue
		}
		for _, v := range values {
			upstreamReq.Header.Add(name, v)
		}
	}
	upstreamReq.ContentLength = int64(len(rewritten))

	logging.Infof("AnthropicPassthrough: complexity=%s model=%s use_1m=%t client_1m=%t upstream=%s bytes=%d",
		label, selectedModel, use1M, clientWants1M, upstreamURL, len(rewritten))

	resp, err := anthropicHTTPClient.Do(upstreamReq)
	if err != nil {
		logging.Errorf("AnthropicPassthrough: upstream call failed: %v", err)
		writeError(w, http.StatusBadGateway, fmt.Sprintf("upstream error: %v", err))
		return
	}
	defer resp.Body.Close()

	for name, values := range resp.Header {
		if _, hop := hopByHopHeaders[http.CanonicalHeaderKey(name)]; hop {
			continue
		}
		for _, v := range values {
			w.Header().Add(name, v)
		}
	}
	w.Header().Set("X-Brick-Selected-Model", selectedModel)
	w.Header().Set("X-Brick-Complexity", label)
	w.WriteHeader(resp.StatusCode)

	flusher, _ := w.(http.Flusher)
	buf := make([]byte, 32*1024)
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, writeErr := w.Write(buf[:n]); writeErr != nil {
				logging.Warnf("AnthropicPassthrough: client write failed: %v", writeErr)
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		if readErr == io.EOF {
			return
		}
		if readErr != nil {
			logging.Warnf("AnthropicPassthrough: upstream read failed: %v", readErr)
			return
		}
	}
}

// requestRequestsContext1M reports whether any incoming Anthropic-Beta header
// value contains a "context-1m-*" flag.
func requestRequestsContext1M(values []string) bool {
	for _, v := range values {
		for _, part := range strings.Split(v, ",") {
			if strings.HasPrefix(strings.TrimSpace(part), "context-1m-") {
				return true
			}
		}
	}
	return false
}

// stripContext1MBeta removes the "context-1m-*" beta flag from a comma-separated
// anthropic-beta header value. The 1M-token context window requires an extra-usage
// paid tier on Opus and is not supported at all on Sonnet/Haiku — forwarding it
// produces "Extra usage is required for 1M context" upstream errors when the
// router downgrades the model. Stripping it falls back to the standard 200K
// context window for all models.
func stripContext1MBeta(v string) string {
	parts := strings.Split(v, ",")
	kept := parts[:0]
	for _, p := range parts {
		t := strings.TrimSpace(p)
		if strings.HasPrefix(t, "context-1m-") {
			continue
		}
		kept = append(kept, t)
	}
	return strings.Join(kept, ",")
}

// extractAnthropicPromptText pulls the classification-relevant text from an
// Anthropic Messages API request body. It concatenates an optional system
// prompt with the text content of the most recent user message, supporting
// both string and structured (array of content blocks) shapes.
func extractAnthropicPromptText(body []byte) string {
	var raw struct {
		System   json.RawMessage `json:"system"`
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &raw); err != nil {
		return ""
	}

	var parts []string
	if sys := decodeAnthropicContent(raw.System); sys != "" {
		parts = append(parts, sys)
	}
	for i := len(raw.Messages) - 1; i >= 0; i-- {
		if raw.Messages[i].Role != "user" {
			continue
		}
		if txt := decodeAnthropicContent(raw.Messages[i].Content); txt != "" {
			parts = append(parts, txt)
			break
		}
	}
	return strings.TrimSpace(strings.Join(parts, "\n"))
}

// decodeAnthropicContent handles the polymorphic Anthropic content field:
// either a JSON string, or an array of content blocks (only "text" blocks are
// extracted; tool_use / image / etc. are ignored for classification purposes).
func decodeAnthropicContent(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var blocks []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return ""
	}
	var texts []string
	for _, b := range blocks {
		if b.Type == "text" && b.Text != "" {
			texts = append(texts, b.Text)
		}
	}
	return strings.Join(texts, "\n")
}

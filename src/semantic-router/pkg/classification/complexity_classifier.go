package classification

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// NvidiaComplexityClassifier calls the external complexity classifier service
// and maps the response directly to a discrete label.
type NvidiaComplexityClassifier struct {
	httpClient *http.Client
	baseURL    string
}

// complexityClassifyResponse is the JSON response from the complexity service.
type complexityClassifyResponse struct {
	Label      string  `json:"label"`
	Confidence float64 `json:"confidence"`
}

// NewNvidiaComplexityClassifier creates a new classifier that calls the complexity service.
func NewNvidiaComplexityClassifier(cfg *config.ComplexityServiceConfig) (*NvidiaComplexityClassifier, error) {
	if cfg.Address == "" {
		return nil, fmt.Errorf("complexity service address is required")
	}
	if cfg.Port == 0 {
		cfg.Port = 8093
	}

	timeout := 5 * time.Second
	if cfg.TimeoutSeconds > 0 {
		timeout = time.Duration(cfg.TimeoutSeconds) * time.Second
	}

	baseURL := fmt.Sprintf("http://%s:%d", cfg.Address, cfg.Port)

	logging.Infof("[ComplexityClassifier] Initialized: endpoint=%s, timeout=%v", baseURL, timeout)

	return &NvidiaComplexityClassifier{
		httpClient: &http.Client{Timeout: timeout},
		baseURL:    baseURL,
	}, nil
}

// Classify sends the query to the complexity service and returns the complexity label.
// Returns a single-element slice like ["complexity:hard"].
// On error or timeout, falls back to ["complexity:medium"].
func (c *NvidiaComplexityClassifier) Classify(query string) ([]string, error) {
	body, err := json.Marshal(map[string]string{"text": query})
	if err != nil {
		return c.fallback("failed to marshal request"), nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), c.httpClient.Timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/classify", bytes.NewReader(body))
	if err != nil {
		return c.fallback("failed to create request"), nil
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return c.fallback(fmt.Sprintf("HTTP request failed: %v", err)), nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return c.fallback(fmt.Sprintf("HTTP status %d", resp.StatusCode)), nil
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return c.fallback("failed to read response body"), nil
	}

	var result complexityClassifyResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return c.fallback(fmt.Sprintf("failed to parse response: %v", err)), nil
	}

	label := "complexity:" + result.Label

	logging.Infof("[ComplexityClassifier] label=%s, confidence=%.4f", label, result.Confidence)

	return []string{label}, nil
}

// fallback logs a warning and returns "complexity:medium" as a safe default.
func (c *NvidiaComplexityClassifier) fallback(reason string) []string {
	logging.Warnf("[ComplexityClassifier] Fallback to medium: %s", reason)
	return []string{"complexity:medium"}
}

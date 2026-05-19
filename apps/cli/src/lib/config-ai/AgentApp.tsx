import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Box, Text, useApp, useInput } from 'ink';
import Spinner from 'ink-spinner';
import TextInput from 'ink-text-input';
import { chatCompletion, type ChatMessageWithTools } from '../client/openai.js';
import { CONFIG_TOOLS, dispatchTool, type ToolContext } from './tools.js';
import { lineDiff, summarizeDiff } from './diff.js';
import { SYSTEM_PROMPT } from './system-prompt.js';

const accent = '#00d4aa';
const danger = '#ff6b6b';

let nextId = 1;

interface UIMessage {
  id: number;
  kind: 'user' | 'assistant' | 'tool' | 'system' | 'patch';
  text: string;
  toolName?: string;
  meta?: string;
}

interface PendingPatch {
  rationale: string;
  oldYaml: string;
  newYaml: string;
  resolve: (accepted: boolean) => void;
}

export interface AgentAppProps {
  profile: string;
  baseUrl: string;
  model: string;
  apiKey?: string;
  configPath: string;
  backendLabel: string;
  maxTokens?: number;
}

export function AgentApp(props: AgentAppProps) {
  const { exit } = useApp();
  const [messages, setMessages] = useState<UIMessage[]>(() => [
    {
      id: nextId++,
      kind: 'system',
      text: `config editor — profile: ${props.profile} · backend: ${props.backendLabel}\nThe agent can read, validate, and patch ${props.configPath}. Press Enter to send, Ctrl+C to exit.`,
    },
  ]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [pending, setPending] = useState<PendingPatch | null>(null);
  const historyRef = useRef<ChatMessageWithTools[]>([
    { role: 'system', content: SYSTEM_PROMPT },
  ]);
  const abortRef = useRef<AbortController | null>(null);

  const append = useCallback((m: Omit<UIMessage, 'id'>) => {
    setMessages((arr) => [...arr, { id: nextId++, ...m }]);
  }, []);

  const confirmPatch: ToolContext['confirmPatch'] = useCallback((rationale, oldYaml, newYaml) => {
    return new Promise((resolve) => {
      setPending({ rationale, oldYaml, newYaml, resolve });
    });
  }, []);

  const ctx: ToolContext = useMemo(() => ({ profile: props.profile, confirmPatch }), [props.profile, confirmPatch]);

  const runAgentTurn = useCallback(async (userText: string) => {
    setBusy(true);
    historyRef.current.push({ role: 'user', content: userText });
    append({ kind: 'user', text: userText });

    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      // Up to 8 tool-call rounds per user turn to prevent runaway loops.
      for (let round = 0; round < 8; round++) {
        if (ctrl.signal.aborted) break;
        const r = await chatCompletion({
          baseUrl: props.baseUrl,
          model: props.model,
          apiKey: props.apiKey,
          messages: historyRef.current,
          tools: CONFIG_TOOLS,
          toolChoice: 'auto',
          maxTokens: props.maxTokens ?? 2048,
        });
        if (r.status >= 400) {
          append({ kind: 'system', text: `error: HTTP ${r.status} — ${(r.content || JSON.stringify(r.raw)).slice(0, 400)}` });
          return;
        }
        if (r.assistantMessage) historyRef.current.push(r.assistantMessage);

        if (r.toolCalls && r.toolCalls.length > 0) {
          if (r.content) append({ kind: 'assistant', text: r.content, meta: r.selectedModel });
          for (const call of r.toolCalls) {
            append({ kind: 'tool', text: `→ ${call.function.name}`, toolName: call.function.name });
            const result = await dispatchTool(call.function.name, call.function.arguments, ctx);
            const payload = result.ok ? result.data : { error: result.error };
            // Tool message back to model
            historyRef.current.push({
              role: 'tool',
              tool_call_id: call.id,
              name: call.function.name,
              content: JSON.stringify(payload).slice(0, 8000),
            });
            // Surface to UI
            if (call.function.name === 'propose_patch') {
              if (payload?.applied) append({ kind: 'system', text: `✓ patch applied to ${payload.path}` });
              else if (payload?.rejected_by_user) append({ kind: 'system', text: '✗ patch rejected' });
              else if (payload?.errors) append({ kind: 'system', text: `validation errors:\n${payload.errors.slice(0, 5).map((e: string) => '  · ' + e).join('\n')}` });
            } else if (call.function.name === 'validate_config') {
              const v = payload?.valid;
              append({ kind: 'system', text: v ? '✓ config valid' : `✗ invalid: ${(payload?.errors ?? []).slice(0, 3).join('; ')}` });
            } else if (call.function.name === 'read_config') {
              if (result.ok) append({ kind: 'system', text: `✓ read config (${payload?.yaml_text?.length ?? 0} chars)` });
              else append({ kind: 'system', text: `✗ read failed: ${result.error}` });
            }
          }
          // Loop again so the model can react to tool results
          continue;
        }
        // No tool calls → final assistant message for this turn.
        if (r.content) append({ kind: 'assistant', text: r.content, meta: r.selectedModel });
        else append({ kind: 'system', text: '(empty response)' });
        return;
      }
      append({ kind: 'system', text: '(stopped: hit max tool-call rounds)' });
    } catch (e: any) {
      append({ kind: 'system', text: `error: ${e?.message ?? String(e)}` });
    } finally {
      abortRef.current = null;
      setBusy(false);
    }
  }, [props.baseUrl, props.model, props.apiKey, props.maxTokens, ctx, append]);

  const onSubmit = useCallback((value: string) => {
    if (!value.trim() || busy || pending) return;
    setInput('');
    runAgentTurn(value);
  }, [busy, pending, runAgentTurn]);

  useInput((inputKey, key) => {
    if (pending) {
      if (key.return || inputKey === 'y' || inputKey === 'Y') { pending.resolve(true); setPending(null); return; }
      if (key.escape || inputKey === 'n' || inputKey === 'N') { pending.resolve(false); setPending(null); return; }
      return;
    }
    if (key.ctrl && inputKey === 'c') {
      if (busy && abortRef.current) abortRef.current.abort();
      else exit();
    }
  });

  return (
    <Box flexDirection="column" paddingX={2} paddingY={1}>
      <Box>
        <Text color={accent} bold>brick config-ai</Text>
        <Text>  · </Text>
        <Text>profile={props.profile}</Text>
        <Text>  · backend={props.backendLabel}</Text>
        {busy && <Text color="yellow">  · thinking…</Text>}
      </Box>

      {messages.map((m) => <MessageView key={m.id} m={m} />)}

      {pending && <PatchView patch={pending} />}

      {!pending && (
        <Box marginTop={1} flexDirection="column">
          <Box>
            <Text color={accent}>{busy ? '⠿ ' : '▷ '}</Text>
            <TextInput
              value={input}
              onChange={setInput}
              onSubmit={onSubmit}
              placeholder={busy ? 'thinking… please wait' : 'describe the change you want…'}
              showCursor
            />
          </Box>
          <Box>
            <Text dimColor>Enter send · Ctrl+C interrupt/exit · y/Enter accept patch · n/Esc reject patch</Text>
          </Box>
        </Box>
      )}
    </Box>
  );
}

function MessageView({ m }: { m: UIMessage }) {
  if (m.kind === 'user') {
    return (
      <Box marginTop={1} flexDirection="column">
        <Text color="cyan">▌ you</Text>
        <Text>{m.text}</Text>
      </Box>
    );
  }
  if (m.kind === 'assistant') {
    return (
      <Box marginTop={1} flexDirection="column">
        <Text color="green" bold>▌ brick{m.meta ? ` (${m.meta})` : ''}</Text>
        <Text>{m.text}</Text>
      </Box>
    );
  }
  if (m.kind === 'tool') {
    return (
      <Box marginTop={1}>
        <Text color={accent} dimColor>{m.text}</Text>
      </Box>
    );
  }
  // system / patch summary
  return (
    <Box marginTop={1}>
      <Text dimColor italic>· {m.text}</Text>
    </Box>
  );
}

function PatchView({ patch }: { patch: PendingPatch }) {
  const lines = useMemo(() => lineDiff(patch.oldYaml, patch.newYaml), [patch.oldYaml, patch.newYaml]);
  const summary = useMemo(() => summarizeDiff(patch.oldYaml, patch.newYaml), [patch.oldYaml, patch.newYaml]);
  const MAX = 80;
  const shown = lines.length > MAX ? lines.slice(0, MAX) : lines;

  return (
    <Box flexDirection="column" borderStyle="round" borderColor={danger} paddingX={1} marginTop={1}>
      <Text color={danger} bold>proposed patch</Text>
      {patch.rationale && <Text dimColor>{patch.rationale}</Text>}
      <Box marginTop={1} flexDirection="column">
        {shown.map((l, i) => <Text key={i}>{l}</Text>)}
        {lines.length > MAX && <Text dimColor>… {lines.length - MAX} more lines truncated</Text>}
      </Box>
      <Box marginTop={1}>
        <Text>  +{summary.added} </Text>
        <Text>-{summary.removed} </Text>
        <Text dimColor>  Apply? </Text>
        <Text color={accent}>[y/Enter] accept</Text>
        <Text dimColor>  ·  </Text>
        <Text color={danger}>[n/Esc] reject</Text>
      </Box>
    </Box>
  );
}

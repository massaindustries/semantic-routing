import React from 'react';
import { Box, Text } from 'ink';
import Spinner from 'ink-spinner';
import type { BablAgentState, BablStatus } from './babl-types.js';

const MAX_CHARS_PER_PANE = 1200;
const MAX_REASONING_CHARS = 400;

function statusColor(s: BablStatus): string | undefined {
  switch (s) {
    case 'streaming': return 'green';
    case 'done': return 'gray';
    case 'error': return 'red';
    case 'aborted': return 'gray';
    case 'idle': return 'gray';
  }
}

function statusGlyph(s: BablStatus): React.ReactNode {
  switch (s) {
    case 'streaming': return <Text color="green"><Spinner type="dots" /></Text>;
    case 'done': return <Text color="green">✓</Text>;
    case 'error': return <Text color="red">✗</Text>;
    case 'aborted': return <Text color="yellow">⊘</Text>;
    case 'idle': return <Text dimColor>·</Text>;
  }
}

export interface BablPaneProps {
  agent: BablAgentState;
  width: number;
  height: number;
  showThinking: boolean;
}

export function BablPane({ agent, width, height, showThinking }: BablPaneProps) {
  const bodyHeight = Math.max(2, height - 5);
  const visible = agent.content.length > MAX_CHARS_PER_PANE
    ? '…' + agent.content.slice(-MAX_CHARS_PER_PANE)
    : agent.content;
  const reasoningVisible = agent.reasoning.length > MAX_REASONING_CHARS
    ? '…' + agent.reasoning.slice(-MAX_REASONING_CHARS)
    : agent.reasoning;
  const reasoningTokApprox = Math.round(agent.reasoning.length / 4);
  const showReasoningInline = showThinking && agent.reasoning.length > 0;
  const showReasoningSummary = !showThinking && agent.reasoning.length > 0 && agent.content.length === 0;

  return (
    <Box
      flexDirection="column"
      borderStyle="round"
      borderColor={statusColor(agent.status)}
      width={width}
      height={height}
      paddingX={2}
      paddingY={0}
    >
      <Box flexShrink={0} marginBottom={1}>
        {statusGlyph(agent.status)}
        <Text> </Text>
        <Text bold>{agent.model}</Text>
      </Box>
      <Box flexGrow={1} overflow="hidden" flexDirection="column" height={bodyHeight}>
        {agent.status === 'error' ? (
          <Text color="red" wrap="wrap">{agent.errorText ?? 'errore'}</Text>
        ) : (
          <>
            {showReasoningInline && (
              <Text color="magenta" dimColor italic wrap="wrap">{reasoningVisible}</Text>
            )}
            {showReasoningSummary && (
              <Text color="magenta" dimColor>
                <Spinner type="dots" /> reasoning… ~{reasoningTokApprox} tok (Ctrl+T)
              </Text>
            )}
            {visible.length > 0 ? (
              <Text wrap="wrap">{visible}</Text>
            ) : (
              !showReasoningInline && !showReasoningSummary && agent.status === 'streaming' && (
                <Text dimColor><Spinner type="dots" /> …</Text>
              )
            )}
          </>
        )}
      </Box>
      <Box flexShrink={0}>
        <Text dimColor>tok: {agent.tokenCount ?? '–'}{agent.reasoning.length > 0 ? ` · think: ~${reasoningTokApprox}` : ''}</Text>
      </Box>
    </Box>
  );
}

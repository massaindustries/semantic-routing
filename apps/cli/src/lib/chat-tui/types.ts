import type { ThinkingMode } from '../client/openai.js';

export interface UserMessage {
  id: number;
  role: 'user';
  content: string;
  ts: number;
}

export interface AssistantMessage {
  id: number;
  role: 'assistant';
  content: string;
  reasoning: string;
  selectedModel?: string;
  thinkingApplied?: string;
  status: 'streaming' | 'done' | 'error' | 'aborted';
  startedAt: number;
  firstTokenAt?: number;
  finishedAt?: number;
  finishReason?: string;
  completionTokens?: number;
  errorText?: string;
}

export interface SystemMessage {
  id: number;
  role: 'system';
  content: string;
}

export type Message = UserMessage | AssistantMessage | SystemMessage;

export interface ChatState {
  messages: Message[];
  showThinking: boolean;
  thinking: ThinkingMode | null;
  stream: boolean;
  queue: string[];      // prompts typed while streaming
  busy: boolean;        // a request is currently in-flight
  thinkingMenuOpen: boolean;
}

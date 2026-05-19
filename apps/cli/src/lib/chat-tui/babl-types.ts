export type BablStatus = 'idle' | 'streaming' | 'done' | 'error' | 'aborted';

export interface BablAgentState {
  model: string;
  status: BablStatus;
  content: string;
  reasoning: string;
  errorText?: string;
  startedAt?: number;
  finishedAt?: number;
  tokenCount?: number;
}

export interface BablTurn {
  index: number;
  agents: Map<string, BablAgentState>;
  previousResponses?: Map<string, string>;
}

export interface BablModeratorState {
  status: BablStatus;
  content: string;
  errorText?: string;
  startedAt?: number;
  finishedAt?: number;
}

export interface BablSession {
  query: string;
  models: string[];
  totalTurns: number;
  currentTurn: number;
  turns: BablTurn[];
  moderator: BablModeratorState;
  globalError?: string;
}

export function makeAgentState(model: string): BablAgentState {
  return { model, status: 'idle', content: '', reasoning: '' };
}

export function makeSession(query: string, models: string[], totalTurns: number): BablSession {
  return {
    query,
    models,
    totalTurns,
    currentTurn: 0,
    turns: [],
    moderator: { status: 'idle', content: '' },
  };
}

export function cloneSession(s: BablSession): BablSession {
  return {
    ...s,
    turns: s.turns.map((t) => ({
      ...t,
      agents: new Map(Array.from(t.agents.entries()).map(([k, v]) => [k, { ...v }])),
      previousResponses: t.previousResponses ? new Map(t.previousResponses) : undefined,
    })),
    moderator: { ...s.moderator },
  };
}

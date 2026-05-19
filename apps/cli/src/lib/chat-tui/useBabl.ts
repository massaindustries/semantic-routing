import { useCallback, useRef, useState } from 'react';
import { chatCompletionStream, type ChatMessage, type ThinkingMode } from '../client/openai.js';
import {
  type BablSession,
  type BablAgentState,
  type BablTurn,
  makeAgentState,
  makeSession,
  cloneSession,
} from './babl-types.js';
import { buildAgentPrompt, buildModeratorPrompt } from './babl-prompts.js';

export interface UseBablOpts {
  agentsBaseUrl: string;
  agentsApiKey?: string;
  models: string[];
  totalTurns: number;
  moderatorModel: string;
  moderatorBaseUrl: string;
  moderatorApiKey?: string;
  maxTokens?: number;
  thinking?: ThinkingMode | null;
}

export interface BablController {
  session: BablSession | null;
  active: boolean;
  runQuery: (query: string, history: ChatMessage[]) => Promise<string | null>;
  abort: () => void;
  reset: () => void;
}

export function useBabl(opts: UseBablOpts): BablController {
  const [session, setSession] = useState<BablSession | null>(null);
  const [active, setActive] = useState(false);
  const abortersRef = useRef<AbortController[]>([]);

  const updateAgent = useCallback((turnIdx: number, model: string, patch: Partial<BablAgentState>) => {
    setSession((prev) => {
      if (!prev) return prev;
      const next = cloneSession(prev);
      const turn = next.turns[turnIdx];
      if (!turn) return prev;
      const cur = turn.agents.get(model);
      if (!cur) return prev;
      turn.agents.set(model, { ...cur, ...patch });
      return next;
    });
  }, []);

  const updateModerator = useCallback((patch: Partial<BablSession['moderator']>) => {
    setSession((prev) => {
      if (!prev) return prev;
      const next = cloneSession(prev);
      next.moderator = { ...next.moderator, ...patch };
      return next;
    });
  }, []);

  const setCurrentTurn = useCallback((turnIdx: number) => {
    setSession((prev) => {
      if (!prev) return prev;
      const next = cloneSession(prev);
      next.currentTurn = turnIdx;
      return next;
    });
  }, []);

  const appendTurn = useCallback((turn: BablTurn) => {
    setSession((prev) => {
      if (!prev) return prev;
      const next = cloneSession(prev);
      next.turns.push({
        ...turn,
        agents: new Map(Array.from(turn.agents.entries()).map(([k, v]) => [k, { ...v }])),
        previousResponses: turn.previousResponses ? new Map(turn.previousResponses) : undefined,
      });
      return next;
    });
  }, []);

  const setGlobalError = useCallback((err: string) => {
    setSession((prev) => {
      if (!prev) return prev;
      const next = cloneSession(prev);
      next.globalError = err;
      return next;
    });
  }, []);

  const streamAgent = useCallback(
    async (
      turnIdx: number,
      turnNumber: number,
      model: string,
      query: string,
      history: ChatMessage[],
      previousResponses: Map<string, string>,
      signal: AbortSignal
    ): Promise<string | null> => {
      updateAgent(turnIdx, model, { status: 'streaming', startedAt: Date.now() });
      const messages = buildAgentPrompt({
        query,
        history,
        turn: turnNumber,
        selfModel: model,
        previousResponses,
      });
      let acc = '';
      let reasoningAcc = '';
      let tokens = 0;
      try {
        const iter = chatCompletionStream({
          baseUrl: opts.agentsBaseUrl,
          model,
          apiKey: opts.agentsApiKey,
          messages,
          maxTokens: opts.maxTokens ?? 1024,
          thinking: opts.thinking ?? null,
          signal,
        });
        for await (const chunk of iter) {
          if (signal.aborted) {
            updateAgent(turnIdx, model, { status: 'aborted', finishedAt: Date.now() });
            return null;
          }
          if (chunk.type === 'reasoning' && chunk.text) {
            reasoningAcc += chunk.text;
            updateAgent(turnIdx, model, { reasoning: reasoningAcc });
          } else if (chunk.type === 'content' && chunk.text) {
            acc += chunk.text;
            tokens++;
            updateAgent(turnIdx, model, { content: acc, tokenCount: tokens });
          } else if (chunk.type === 'error') {
            updateAgent(turnIdx, model, {
              status: 'error',
              errorText: `${chunk.status ?? '?'}: ${(chunk.error ?? '').slice(0, 120)}`,
              finishedAt: Date.now(),
            });
            return null;
          } else if (chunk.type === 'done') {
            updateAgent(turnIdx, model, { status: 'done', finishedAt: Date.now(), tokenCount: tokens });
          }
        }
        return acc.trim() || null;
      } catch (e: any) {
        if (signal.aborted) {
          updateAgent(turnIdx, model, { status: 'aborted', finishedAt: Date.now() });
        } else {
          updateAgent(turnIdx, model, {
            status: 'error',
            errorText: (e?.message ?? String(e)).slice(0, 120),
            finishedAt: Date.now(),
          });
        }
        return null;
      }
    },
    [opts.agentsBaseUrl, opts.agentsApiKey, opts.maxTokens, opts.thinking, updateAgent]
  );

  const streamModerator = useCallback(
    async (query: string, finalResponses: Map<string, string>, signal: AbortSignal): Promise<string | null> => {
      updateModerator({ status: 'streaming', startedAt: Date.now() });
      let acc = '';
      try {
        const iter = chatCompletionStream({
          baseUrl: opts.moderatorBaseUrl,
          model: opts.moderatorModel,
          messages: buildModeratorPrompt(query, finalResponses),
          apiKey: opts.moderatorApiKey,
          maxTokens: opts.maxTokens ?? 2048,
          thinking: null,
          signal,
        });
        for await (const chunk of iter) {
          if (signal.aborted) {
            updateModerator({ status: 'aborted', finishedAt: Date.now() });
            return null;
          }
          if (chunk.type === 'content' && chunk.text) {
            acc += chunk.text;
            updateModerator({ content: acc });
          } else if (chunk.type === 'error') {
            updateModerator({
              status: 'error',
              errorText: `${chunk.status ?? '?'}: ${(chunk.error ?? '').slice(0, 200)}`,
              finishedAt: Date.now(),
            });
            return null;
          } else if (chunk.type === 'done') {
            updateModerator({ status: 'done', finishedAt: Date.now() });
          }
        }
        return acc.trim() || null;
      } catch (e: any) {
        if (signal.aborted) updateModerator({ status: 'aborted', finishedAt: Date.now() });
        else updateModerator({ status: 'error', errorText: (e?.message ?? String(e)).slice(0, 200), finishedAt: Date.now() });
        return null;
      }
    },
    [opts.moderatorBaseUrl, opts.moderatorModel, opts.moderatorApiKey, opts.maxTokens, updateModerator]
  );

  const runQuery = useCallback(
    async (query: string, history: ChatMessage[]): Promise<string | null> => {
      const initial = makeSession(query, opts.models, opts.totalTurns);
      setSession(initial);
      setActive(true);

      let lastResponses = new Map<string, string>();

      try {
        for (let t = 1; t <= opts.totalTurns; t++) {
          const turn: BablTurn = {
            index: t,
            agents: new Map(opts.models.map((m) => [m, makeAgentState(m)])),
            previousResponses: new Map(lastResponses),
          };
          appendTurn(turn);
          setCurrentTurn(t);

          const aborters = opts.models.map(() => new AbortController());
          abortersRef.current = aborters;
          const turnIdx = t - 1;

          const promises = opts.models.map((m, i) =>
            streamAgent(turnIdx, t, m, query, history, lastResponses, aborters[i].signal)
          );
          const results = await Promise.allSettled(promises);

          const newResponses = new Map<string, string>();
          results.forEach((r, i) => {
            if (r.status === 'fulfilled' && r.value && r.value.trim()) {
              newResponses.set(opts.models[i], r.value);
            }
          });

          if (aborters.some((a) => a.signal.aborted)) {
            setActive(false);
            return null;
          }

          if (newResponses.size === 0) {
            setGlobalError(`tutti gli agenti hanno fallito al turno ${t}`);
            setActive(false);
            return null;
          }
          lastResponses = newResponses;
        }

        const modAbort = new AbortController();
        abortersRef.current = [modAbort];
        const synth = await streamModerator(query, lastResponses, modAbort.signal);
        setActive(false);
        return synth;
      } catch (e: any) {
        setGlobalError((e?.message ?? String(e)).slice(0, 200));
        setActive(false);
        return null;
      }
    },
    [opts.models, opts.totalTurns, appendTurn, setCurrentTurn, setGlobalError, streamAgent, streamModerator]
  );

  const abort = useCallback(() => {
    for (const a of abortersRef.current) {
      try { a.abort(); } catch {}
    }
  }, []);

  const reset = useCallback(() => {
    abort();
    setSession(null);
    setActive(false);
  }, [abort]);

  return { session, active, runQuery, abort, reset };
}

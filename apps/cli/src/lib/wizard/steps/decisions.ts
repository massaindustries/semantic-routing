import * as p from '@clack/prompts';
import type { Rule, Condition } from '../../config/schema.js';

export async function runDecisionBuilder(availableModels: string[]): Promise<any[]> {
  const decisions: any[] = [];
  for (;;) {
    const more = decisions.length === 0
      ? true
      : await p.confirm({ message: `Add another decision? (current: ${decisions.length})`, initialValue: false });
    if (p.isCancel(more)) { p.cancel('aborted'); process.exit(0); }
    if (!more) break;

    const name = await p.text({ message: 'Decision name (e.g. coding_easy):', validate: (v) => v ? undefined : 'required' });
    if (p.isCancel(name)) { p.cancel('aborted'); process.exit(0); }
    const desc = await p.text({ message: 'Description:', placeholder: 'optional' });
    if (p.isCancel(desc)) { p.cancel('aborted'); process.exit(0); }

    const rules = await buildRule();

    const modelChoice = await p.select({
      message: 'Model to route to:',
      options: availableModels.map((m) => ({ value: m, label: m })),
    });
    if (p.isCancel(modelChoice)) { p.cancel('aborted'); process.exit(0); }
    const useReasoning = await p.confirm({ message: 'Use reasoning?', initialValue: false });
    if (p.isCancel(useReasoning)) { p.cancel('aborted'); process.exit(0); }
    const ref: any = { model: String(modelChoice), use_reasoning: !!useReasoning };
    if (useReasoning) {
      const eff = await p.select({
        message: 'Reasoning effort:',
        options: [
          { value: 'low', label: 'low' },
          { value: 'medium', label: 'medium' },
          { value: 'high', label: 'high' },
        ],
        initialValue: 'medium',
      });
      if (p.isCancel(eff)) { p.cancel('aborted'); process.exit(0); }
      ref.reasoning_effort = String(eff);
    }
    decisions.push({ name: String(name), description: String(desc || ''), rules, modelRefs: [ref] });
  }
  if (decisions.length === 0) {
    p.note('no decisions defined — router will always use default_model', 'warning');
  }
  return decisions;
}

async function buildRule(depth = 0): Promise<Rule> {
  const kind = await p.select({
    message: depth === 0 ? 'Top-level rule:' : `  └ subrule (depth ${depth}):`,
    options: [
      { value: 'AND', label: 'AND (all conditions match)' },
      { value: 'OR', label: 'OR (any condition matches)' },
      { value: 'NOT', label: 'NOT (negate one condition)' },
      { value: 'condition', label: 'leaf condition (keyword / domain / complexity)' },
    ],
  });
  if (p.isCancel(kind)) { p.cancel('aborted'); process.exit(0); }

  if (kind === 'condition') return await buildCondition();

  const conditions: (Rule | Condition)[] = [];
  const max = kind === 'NOT' ? 1 : 5;
  for (let i = 0; i < max; i++) {
    const stop = i > 0 && (kind !== 'NOT')
      ? await p.confirm({ message: `Add another sub-condition to ${kind}?`, initialValue: i < 1 })
      : false;
    if (i > 0 && !stop) break;
    if (i > 0 && p.isCancel(stop)) { p.cancel('aborted'); process.exit(0); }
    const sub = await p.select({
      message: `${kind} child #${i + 1}:`,
      options: [
        { value: 'leaf', label: 'leaf condition' },
        { value: 'group', label: 'nested AND/OR/NOT group' },
      ],
      initialValue: 'leaf',
    });
    if (p.isCancel(sub)) { p.cancel('aborted'); process.exit(0); }
    if (sub === 'leaf') conditions.push(await buildCondition());
    else conditions.push(await buildRule(depth + 1));
    if (kind === 'NOT') break;
  }
  return { operator: kind as 'AND' | 'OR' | 'NOT', conditions };
}

async function buildCondition(): Promise<Condition> {
  const t = await p.select({
    message: 'condition type:',
    options: [
      { value: 'keyword', label: 'keyword (refs keyword_rules entry by name)' },
      { value: 'domain', label: 'domain (e.g. "computer science")' },
      { value: 'complexity', label: 'complexity:easy/medium/hard' },
    ],
  });
  if (p.isCancel(t)) { p.cancel('aborted'); process.exit(0); }
  let name = '';
  if (t === 'keyword') {
    const v = await p.text({ message: 'keyword rule name:', placeholder: 'code_keywords', defaultValue: 'code_keywords' });
    if (p.isCancel(v)) { p.cancel('aborted'); process.exit(0); }
    name = String(v || 'code_keywords');
  } else if (t === 'domain') {
    const v = await p.text({ message: 'domain name:', placeholder: 'computer science' });
    if (p.isCancel(v)) { p.cancel('aborted'); process.exit(0); }
    name = String(v);
  } else {
    const v = await p.select({
      message: 'complexity level:',
      options: [
        { value: 'complexity:easy', label: 'easy' },
        { value: 'complexity:medium', label: 'medium' },
        { value: 'complexity:hard', label: 'hard' },
      ],
    });
    if (p.isCancel(v)) { p.cancel('aborted'); process.exit(0); }
    name = String(v);
  }
  return { type: t as 'keyword' | 'domain' | 'complexity', name };
}

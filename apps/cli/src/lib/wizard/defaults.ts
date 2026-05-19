import type { Decision, Rule } from './types.js';

export function defaultDecisions(opts: {
  codingEasyModel: string;
  codingHardModel: string;
  generalEasyModel: string;
  generalMediumModel: string;
  generalHardModel: string;
  codingHardReasoningFamily?: string;
  generalHardReasoningFamily?: string;
}): any[] {
  const codingTrigger = (extra: Rule | null = null): Rule => ({
    operator: 'OR',
    conditions: [
      { type: 'keyword', name: 'code_keywords' },
      { type: 'domain', name: 'computer science' },
    ],
  });

  return [
    {
      name: 'coding_easy',
      description: 'Coding easy/medium → coder model',
      rules: {
        operator: 'AND',
        conditions: [
          codingTrigger(),
          {
            operator: 'OR',
            conditions: [
              { type: 'complexity', name: 'complexity:easy' },
              { type: 'complexity', name: 'complexity:medium' },
            ],
          },
        ],
      },
      modelRefs: [{ model: opts.codingEasyModel, use_reasoning: false }],
    },
    {
      name: 'coding_hard',
      description: 'Coding hard → reasoning model',
      rules: {
        operator: 'AND',
        conditions: [codingTrigger(), { type: 'complexity', name: 'complexity:hard' }],
      },
      modelRefs: [
        {
          model: opts.codingHardModel,
          use_reasoning: !!opts.codingHardReasoningFamily,
          ...(opts.codingHardReasoningFamily ? { reasoning_effort: 'low' as const } : {}),
        },
      ],
    },
    {
      name: 'general_easy',
      description: 'General easy → small model',
      rules: {
        operator: 'AND',
        conditions: [
          { operator: 'NOT', conditions: [codingTrigger()] },
          { type: 'complexity', name: 'complexity:easy' },
        ],
      },
      modelRefs: [{ model: opts.generalEasyModel, use_reasoning: false }],
    },
    {
      name: 'general_medium',
      description: 'General medium → mid model',
      rules: {
        operator: 'AND',
        conditions: [
          { operator: 'NOT', conditions: [codingTrigger()] },
          { type: 'complexity', name: 'complexity:medium' },
        ],
      },
      modelRefs: [{ model: opts.generalMediumModel, use_reasoning: false }],
    },
    {
      name: 'general_hard',
      description: 'General hard → reasoning model',
      rules: {
        operator: 'AND',
        conditions: [
          { operator: 'NOT', conditions: [codingTrigger()] },
          { type: 'complexity', name: 'complexity:hard' },
        ],
      },
      modelRefs: [
        {
          model: opts.generalHardModel,
          use_reasoning: !!opts.generalHardReasoningFamily,
          ...(opts.generalHardReasoningFamily ? { reasoning_effort: 'low' as const } : {}),
        },
      ],
    },
  ];
}

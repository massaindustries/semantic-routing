// Benchmark: does any method actually reduce latency on qwen3.5-9b via Regolo?
// Methods tested:
//   A) baseline                                   — no flag, default thinking ON
//   B) extra_body chat_template_kwargs            — hard switch enable_thinking=false
//   C) /no_think in user prompt                   — soft switch
//   D) /no_think in system prompt                 — soft switch (system)
// Each method × each prompt × N samples. Reports mean latency, mean total_tokens, mean reasoning chars.

import { readFileSync } from 'node:fs';
import { paths } from '../src/lib/config/paths.js';

const MODEL = 'qwen3.5-9b';
const ENDPOINT = 'https://api.regolo.ai/v1/chat/completions';
const SAMPLES = Number(process.env.SAMPLES ?? 4);
const KEY = (() => {
  const m = readFileSync(paths.env, 'utf8').match(/^REGOLO_API_KEY=(.+)$/m);
  if (!m) throw new Error('REGOLO_API_KEY not in ~/.brick/.env');
  return m[1].trim();
})();

const PROMPTS = [
  'translate hello to french',
  'what is 2+2',
  'name the capital of italy',
  'write a one-line python function that doubles a number',
  'summarize photosynthesis in one sentence',
];

interface Method { name: string; build: (user: string) => any; }

const METHODS: Method[] = [
  {
    name: 'A_baseline',
    build: (user) => ({
      model: MODEL,
      messages: [{ role: 'user', content: user }],
      max_tokens: 512,
    }),
  },
  {
    name: 'B_extra_body_enable_thinking_false',
    build: (user) => ({
      model: MODEL,
      messages: [{ role: 'user', content: user }],
      max_tokens: 512,
      chat_template_kwargs: { enable_thinking: false },
    }),
  },
  {
    name: 'C_user_no_think',
    build: (user) => ({
      model: MODEL,
      messages: [{ role: 'user', content: user + ' /no_think' }],
      max_tokens: 512,
    }),
  },
  {
    name: 'D_system_no_think',
    build: (user) => ({
      model: MODEL,
      messages: [
        { role: 'system', content: '/no_think' },
        { role: 'user', content: user },
      ],
      max_tokens: 512,
    }),
  },
];

interface Sample { latencyMs: number; totalTokens: number; completionTokens: number; reasoningChars: number; contentChars: number; }
interface Stats { method: string; samples: Sample[]; meanLat: number; medLat: number; meanTok: number; meanReasoningChars: number; meanContentChars: number; }

async function callOnce(body: any): Promise<Sample | null> {
  const t0 = performance.now();
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 60000);
  try {
    const r = await fetch(ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${KEY}` },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    const j: any = await r.json().catch(() => ({}));
    if (!r.ok) {
      console.error(`  http ${r.status}: ${JSON.stringify(j).slice(0, 120)}`);
      return null;
    }
    const lat = performance.now() - t0;
    const msg = j?.choices?.[0]?.message ?? {};
    return {
      latencyMs: lat,
      totalTokens: j?.usage?.total_tokens ?? 0,
      completionTokens: j?.usage?.completion_tokens ?? 0,
      reasoningChars: (msg.reasoning_content ?? '').length,
      contentChars: (msg.content ?? '').length,
    };
  } finally { clearTimeout(t); }
}

function mean(xs: number[]): number { return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0; }
function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

(async () => {
  console.log(`benchmark on ${MODEL} via Regolo direct, ${SAMPLES} samples × ${PROMPTS.length} prompts × ${METHODS.length} methods`);
  console.log(`(serial calls; warm-up = first call discarded)\n`);

  const all: Stats[] = [];

  for (const method of METHODS) {
    const samples: Sample[] = [];
    process.stdout.write(`[${method.name.padEnd(36)}] `);
    let warmup = true;
    for (const prompt of PROMPTS) {
      for (let i = 0; i < SAMPLES; i++) {
        const s = await callOnce(method.build(prompt));
        if (s === null) { process.stdout.write('X'); continue; }
        if (warmup) { warmup = false; process.stdout.write('w'); continue; }
        samples.push(s);
        process.stdout.write('.');
      }
    }
    process.stdout.write('\n');
    const lat = samples.map((s) => s.latencyMs);
    all.push({
      method: method.name,
      samples,
      meanLat: mean(lat),
      medLat: median(lat),
      meanTok: mean(samples.map((s) => s.completionTokens)),
      meanReasoningChars: mean(samples.map((s) => s.reasoningChars)),
      meanContentChars: mean(samples.map((s) => s.contentChars)),
    });
  }

  console.log('\n=== summary ===');
  console.log('method                                |  n |  mean lat |  median lat |  mean tok |  reasoning chars |  content chars');
  console.log('--------------------------------------+----+-----------+-------------+-----------+------------------+----------------');
  const baseline = all.find((s) => s.method === 'A_baseline');
  for (const s of all) {
    const delta = baseline ? `${s.meanLat < baseline.meanLat ? '-' : '+'}${Math.abs(((s.meanLat - baseline.meanLat) / baseline.meanLat) * 100).toFixed(1)}%` : '';
    console.log(
      `${s.method.padEnd(38)}|${String(s.samples.length).padStart(4)}|${(s.meanLat / 1000).toFixed(2).padStart(8)}s ${delta.padStart(8)} | ${(s.medLat / 1000).toFixed(2).padStart(8)}s | ${s.meanTok.toFixed(0).padStart(8)} | ${s.meanReasoningChars.toFixed(0).padStart(15)}  | ${s.meanContentChars.toFixed(0).padStart(13)}`
    );
  }

  console.log('\n=== verdict ===');
  if (!baseline || baseline.samples.length === 0) { console.log('baseline failed — cannot compute verdict'); return; }
  for (const s of all) {
    if (s.method === 'A_baseline') continue;
    const speedup = (baseline.meanLat - s.meanLat) / baseline.meanLat;
    const reasoningReduction = baseline.meanReasoningChars > 0
      ? (baseline.meanReasoningChars - s.meanReasoningChars) / baseline.meanReasoningChars
      : 0;
    let verdict = '';
    if (speedup > 0.3) verdict = 'CLEAR WIN — faster';
    else if (speedup > 0.1) verdict = 'modest improvement';
    else if (speedup > -0.1) verdict = 'no significant difference';
    else verdict = 'SLOWER';
    if (s.meanReasoningChars === 0 && baseline.meanReasoningChars > 0) verdict += ' · reasoning OFF';
    else if (reasoningReduction > 0.5) verdict += ` · reasoning -${(reasoningReduction * 100).toFixed(0)}%`;
    console.log(`${s.method.padEnd(38)}: ${verdict}`);
  }
})();

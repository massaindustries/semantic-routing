import { Command, Flags } from '@oclif/core';

type DiagClassifier = {
  enabled: boolean;
  endpoint?: string;
  reachable?: boolean;
  device?: string;
  model?: string;
  latency_ms?: number;
  error?: string;
};

type ParsedMetrics = {
  requestsByLabelModel: Map<string, number>;
  fallbackTotal: number;
  classifyDurationCount: number;
  classifyDurationSum: number;
  classifyDurationBuckets: Array<{ le: number; count: number }>;
};

const COLORS = {
  reset: '\x1b[0m',
  bold: '\x1b[1m',
  dim: '\x1b[2m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  cyan: '\x1b[36m',
};

const tick = `${COLORS.green}✓${COLORS.reset}`;
const cross = `${COLORS.red}✗${COLORS.reset}`;

export default class ClaudeStatus extends Command {
  static description =
    'Show whether Claude Code is wired to the local Brick router, and how it has been routing prompts.';

  static examples = [
    '<%= config.bin %> claude status',
    '<%= config.bin %> claude status --url http://localhost:19000',
  ];

  static flags = {
    url: Flags.string({
      char: 'u',
      description: 'brick base URL (default: $ANTHROPIC_BASE_URL or http://localhost:18000)',
    }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(ClaudeStatus);
    const envUrl = process.env.ANTHROPIC_BASE_URL?.trim();
    const baseUrl = (flags.url ?? envUrl ?? 'http://localhost:18000').replace(/\/$/, '');

    this.log('');
    this.log(`${COLORS.bold}${COLORS.cyan}Connection${COLORS.reset}`);
    this.log(
      `  ANTHROPIC_BASE_URL  ${envUrl ?? `${COLORS.dim}(not set)${COLORS.reset}`}  ${
        envUrl === baseUrl ? `${tick} attached` : `${cross} not attached — set ANTHROPIC_BASE_URL=${baseUrl}`
      }`
    );

    const health = await probeHealth(baseUrl);
    this.log(`  brick             ${baseUrl}  ${health ? `${tick} healthy` : `${cross} unreachable`}`);

    const diag = await fetchDiag(baseUrl);
    if (!diag) {
      this.log(`  classifier          ${cross} could not reach /api/v1/diag/classifier on brick`);
    } else if (!diag.enabled) {
      this.log(`  classifier          ${COLORS.dim}disabled in config${COLORS.reset}`);
    } else if (diag.reachable) {
      this.log(
        `  classifier          ${diag.endpoint ?? '(unknown)'}  ${tick} healthy (${diag.device ?? '?'}, last ${
          diag.latency_ms ?? '?'
        }ms)`
      );
    } else {
      this.log(
        `  classifier          ${diag.endpoint ?? '(unknown)'}  ${cross} unreachable${
          diag.error ? ` — ${diag.error}` : ''
        }`
      );
    }

    if (!health) {
      this.log('');
      this.log(`${COLORS.yellow}brick is not responding. Skipping routing stats.${COLORS.reset}`);
      return;
    }

    const metrics = await fetchMetrics(baseUrl);
    if (!metrics) {
      this.log('');
      this.log(`${COLORS.yellow}Metrics endpoint not reachable; cannot show routing stats.${COLORS.reset}`);
      this.log(`${COLORS.dim}  (brick exposes Prometheus metrics on its own port — usually 9190 or 19190.)${COLORS.reset}`);
      return;
    }

    this.log('');
    this.log(`${COLORS.bold}${COLORS.cyan}Routing since restart${COLORS.reset}`);
    this.printRoutingStats(metrics);

    this.log('');
    this.log(`${COLORS.dim}Stop:    docker compose -f docker-compose.brick-cc.yml down${COLORS.reset}`);
    this.log(`${COLORS.dim}Logs:    docker compose -f docker-compose.brick-cc.yml logs -f${COLORS.reset}`);
  }

  private printRoutingStats(m: ParsedMetrics): void {
    const total = [...m.requestsByLabelModel.values()].reduce((a, b) => a + b, 0);
    if (total === 0) {
      this.log(`  ${COLORS.dim}No /v1/messages requests served yet.${COLORS.reset}`);
      return;
    }

    const labelOrder = ['easy', 'medium', 'hard'];
    type Row = { label: string; model: string; count: number };
    const rows: Row[] = [];
    for (const [key, count] of m.requestsByLabelModel.entries()) {
      const [label, model] = key.split('|');
      rows.push({ label, model, count });
    }
    rows.sort((a, b) => labelOrder.indexOf(a.label) - labelOrder.indexOf(b.label));

    this.log(`  Total requests        ${total}`);
    for (const row of rows) {
      const pct = ((row.count / total) * 100).toFixed(0).padStart(2);
      this.log(`  ${row.label.padEnd(6)} → ${row.model.padEnd(20)}  ${String(row.count).padStart(4)}  (${pct}%)`);
    }

    const p50 = bucketPercentile(m.classifyDurationBuckets, m.classifyDurationCount, 0.5);
    const p95 = bucketPercentile(m.classifyDurationBuckets, m.classifyDurationCount, 0.95);
    if (p50 !== null && p95 !== null) {
      this.log(`  Classifier p50/p95    ${formatLatency(p50)} / ${formatLatency(p95)}`);
    }

    const fallbackPct = total === 0 ? 0 : (m.fallbackTotal / total) * 100;
    const fallbackColor = fallbackPct > 5 ? COLORS.red : fallbackPct > 1 ? COLORS.yellow : COLORS.green;
    this.log(`  Fallback rate          ${fallbackColor}${fallbackPct.toFixed(1)}%${COLORS.reset} (${m.fallbackTotal} fallbacks)`);
  }
}

async function probeHealth(baseUrl: string): Promise<boolean> {
  try {
    const r = await fetch(`${baseUrl}/health`, { signal: AbortSignal.timeout(2000) });
    return r.ok;
  } catch {
    return false;
  }
}

async function fetchDiag(baseUrl: string): Promise<DiagClassifier | null> {
  try {
    const r = await fetch(`${baseUrl}/api/v1/diag/classifier`, { signal: AbortSignal.timeout(4000) });
    if (!r.ok) return null;
    return (await r.json()) as DiagClassifier;
  } catch {
    return null;
  }
}

async function fetchMetrics(baseUrl: string): Promise<ParsedMetrics | null> {
  // The metrics endpoint is exposed on a separate port — try the same host on
  // the typical metrics ports first, then fall back to the proxy port itself.
  const url = new URL(baseUrl);
  const candidates = [
    `${url.protocol}//${url.hostname}:19190/metrics`,
    `${url.protocol}//${url.hostname}:9190/metrics`,
    `${baseUrl}/metrics`,
  ];
  for (const u of candidates) {
    try {
      const r = await fetch(u, { signal: AbortSignal.timeout(3000) });
      if (r.ok) return parsePromExposition(await r.text());
    } catch {
      // try next
    }
  }
  return null;
}

function parsePromExposition(body: string): ParsedMetrics {
  const out: ParsedMetrics = {
    requestsByLabelModel: new Map(),
    fallbackTotal: 0,
    classifyDurationCount: 0,
    classifyDurationSum: 0,
    classifyDurationBuckets: [],
  };

  for (const raw of body.split('\n')) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;

    if (line.startsWith('brick_cc_requests_total')) {
      const m = line.match(/^brick_cc_requests_total\{([^}]*)\}\s+([0-9.eE+-]+)$/);
      if (!m) continue;
      const labels = parseLabels(m[1]);
      const key = `${labels.label ?? 'unknown'}|${labels.model ?? 'unknown'}`;
      out.requestsByLabelModel.set(key, (out.requestsByLabelModel.get(key) ?? 0) + Number(m[2]));
    } else if (line.startsWith('brick_cc_classify_fallback_total')) {
      const m = line.match(/^brick_cc_classify_fallback_total\s+([0-9.eE+-]+)$/);
      if (m) out.fallbackTotal += Number(m[1]);
    } else if (line.startsWith('brick_cc_classify_duration_seconds_count')) {
      const m = line.match(/^brick_cc_classify_duration_seconds_count\s+([0-9.eE+-]+)$/);
      if (m) out.classifyDurationCount += Number(m[1]);
    } else if (line.startsWith('brick_cc_classify_duration_seconds_sum')) {
      const m = line.match(/^brick_cc_classify_duration_seconds_sum\s+([0-9.eE+-]+)$/);
      if (m) out.classifyDurationSum += Number(m[1]);
    } else if (line.startsWith('brick_cc_classify_duration_seconds_bucket')) {
      const m = line.match(/^brick_cc_classify_duration_seconds_bucket\{le="([^"]+)"\}\s+([0-9.eE+-]+)$/);
      if (!m) continue;
      const le = m[1] === '+Inf' ? Number.POSITIVE_INFINITY : Number(m[1]);
      out.classifyDurationBuckets.push({ le, count: Number(m[2]) });
    }
  }

  out.classifyDurationBuckets.sort((a, b) => a.le - b.le);
  return out;
}

function parseLabels(s: string): Record<string, string> {
  const out: Record<string, string> = {};
  // Naive label parser; assumes well-formed Prometheus exposition output.
  for (const m of s.matchAll(/(\w+)="([^"]*)"/g)) {
    out[m[1]] = m[2];
  }
  return out;
}

function bucketPercentile(
  buckets: Array<{ le: number; count: number }>,
  total: number,
  q: number
): number | null {
  if (total === 0 || buckets.length === 0) return null;
  const target = total * q;
  for (const b of buckets) {
    if (b.count >= target) return b.le;
  }
  return buckets[buckets.length - 1].le;
}

function formatLatency(seconds: number): string {
  if (!isFinite(seconds)) return '>10s';
  const ms = seconds * 1000;
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.round(ms)}ms`;
}

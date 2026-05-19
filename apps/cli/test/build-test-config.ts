// Programmatic config builder for testing — bypasses interactive wizard.
import { readFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import yaml from 'js-yaml';
import { ConfigSchema, type BrickConfig } from '../src/lib/config/schema.js';
import { saveConfig } from '../src/lib/config/save.js';
import { defaultDecisions } from '../src/lib/wizard/defaults.js';
import { writeCompose } from '../src/lib/docker/compose.js';
import { catalog, reasoningFamiliesDefault } from '../src/lib/catalog/index.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const TEMPLATE_DIR = join(__dirname, '..', 'templates');

export interface BuildOpts {
  providers: ('regolo' | 'openai' | 'local')[];
  withClassifier?: boolean;
  withComplexity?: boolean;
  withMultimodal?: boolean;
  port?: number;
}

export async function buildConfig(opts: BuildOpts): Promise<BrickConfig> {
  const providers: Record<string, any> = {};
  const providerProfiles: Record<string, any> = {};
  const providerEndpoints: any[] = [];
  const modelConfig: Record<string, any> = {};
  const reasoningFamilies: Record<string, any> = {};

  for (const pid of opts.providers) {
    const cat = catalog[pid];
    providers[pid] = { type: 'openai_compatible', base_url: cat.base_url };
    providerProfiles[pid] = { type: 'openai_compatible', base_url: cat.base_url };
    providerEndpoints.push({ name: pid, provider_profile: pid, weight: 1 });
    for (const m of cat.models) {
      modelConfig[m.id] = {
        preferred_endpoints: [pid],
        param_size: m.param_size,
        ...(m.reasoning_family ? { reasoning_family: m.reasoning_family } : {}),
      };
      if (m.reasoning_family && (reasoningFamiliesDefault as any)[m.reasoning_family]) {
        reasoningFamilies[m.reasoning_family] = (reasoningFamiliesDefault as any)[m.reasoning_family];
      }
    }
  }

  const hasRegolo = opts.providers.includes('regolo');
  const codingEasy = hasRegolo ? 'qwen3-coder-next' : 'gpt-4o-mini';
  const codingHard = hasRegolo ? 'minimax-m2.5' : 'gpt-4o';
  const generalEasy = hasRegolo ? 'qwen3.5-9b' : 'gpt-4o-mini';
  const generalMed = hasRegolo ? 'qwen3.5-122b' : 'gpt-4.1';
  const generalHard = hasRegolo ? 'minimax-m2.5' : 'gpt-4o';
  const codingHardFam = modelConfig[codingHard]?.reasoning_family;
  const generalHardFam = modelConfig[generalHard]?.reasoning_family;

  const decisions = defaultDecisions({
    codingEasyModel: codingEasy,
    codingHardModel: codingHard,
    generalEasyModel: generalEasy,
    generalMediumModel: generalMed,
    generalHardModel: generalHard,
    codingHardReasoningFamily: codingHardFam,
    generalHardReasoningFamily: generalHardFam,
  });

  const tplKw = await readFile(join(TEMPLATE_DIR, 'keywords.default.yaml'), 'utf8');
  const keywordRules = yaml.load(tplKw) as any[];

  const cfg: BrickConfig = ConfigSchema.parse({
    model: { name: 'brick', description: 'Virtual multimodal routing model' },
    providers,
    brick: opts.withMultimodal ? {
      enabled: true,
      stt_model: 'faster-whisper-large-v3',
      stt_endpoint: 'https://api.regolo.ai/v1/audio/transcriptions',
      ocr_model: 'deepseek-ocr-2',
      ocr_endpoint: 'https://api.regolo.ai/v1/chat/completions',
      vision_model: 'qwen3.5-122b',
      vision_endpoint: 'https://api.regolo.ai/v1/chat/completions',
      ocr_min_text_length: 10,
    } : undefined,
    server_port: opts.port ?? 8000,
    auto_model_name: 'brick',
    provider_profiles: providerProfiles,
    provider_endpoints: providerEndpoints,
    default_model: hasRegolo ? 'qwen3.5-122b' : 'gpt-4o',
    model_config: modelConfig,
    reasoning_families: reasoningFamilies,
    default_reasoning_effort: 'medium',
    classifier: opts.withClassifier ? {
      category_model: {
        model_id: 'models/mom-domain-classifier',
        use_modernbert: true,
        threshold: 0.45,
        use_cpu: true,
        category_mapping_path: 'models/mom-domain-classifier/category_mapping.json',
      },
    } : undefined,
    complexity_service: opts.withComplexity ? {
      enabled: true, address: '172.19.0.1', port: 8094, timeout_seconds: 5,
    } : undefined,
    keyword_rules: keywordRules,
    decisions,
  });

  await saveConfig(cfg);
  await writeCompose({ port: cfg.server_port });
  return cfg;
}

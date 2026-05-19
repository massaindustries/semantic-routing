import { Command, Args, Flags } from '@oclif/core';
import * as p from '@clack/prompts';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';
import { loadConfigRaw } from '../../lib/config/load.js';
import { ConfigSchema } from '../../lib/config/schema.js';
import { saveConfig } from '../../lib/config/save.js';
import { paths, resolveProfile } from '../../lib/config/paths.js';
import { catalog } from '../../lib/catalog/index.js';
import { ok, err } from '../../lib/ui/banners.js';

export default class AddProvider extends Command {
  static description = 'Add a provider to ~/.brick/config.yaml';
  static args = { id: Args.string({ required: true, description: 'provider id (regolo|openai|local|<custom>)' }) };
  static flags = {
    'base-url': Flags.string({ description: 'override base_url' }),
    'api-key': Flags.string({ description: 'API key (saved in .env, never in YAML)' }),
  };
  async run(): Promise<void> {
    const { args, flags } = await this.parse(AddProvider);
    const raw = (await loadConfigRaw()) as any;
    const cat = catalog[args.id];
    const baseUrl = flags['base-url'] ?? cat?.base_url;
    if (!baseUrl) { err('base-url required for unknown provider'); this.exit(1); }
    raw.providers = raw.providers ?? {};
    raw.providers[args.id] = { type: 'openai_compatible', base_url: baseUrl };
    raw.provider_profiles = raw.provider_profiles ?? {};
    raw.provider_profiles[args.id] = { type: 'openai_compatible', base_url: baseUrl };
    raw.provider_endpoints = raw.provider_endpoints ?? [];
    if (!raw.provider_endpoints.find((v: any) => v.name === args.id)) {
      raw.provider_endpoints.push({ name: args.id, provider_profile: args.id, weight: 1 });
    }
    const cfg = ConfigSchema.parse(raw);
    await saveConfig(cfg);
    const envKey = cat?.env_key ?? `${args.id.toUpperCase()}_API_KEY`;
    let key = flags['api-key'];
    if (!key) {
      const v = await p.password({ message: `${envKey} (saved to .env):` });
      if (p.isCancel(v)) { p.cancel('aborted'); process.exit(0); }
      key = String(v);
    }
    await mergeEnv(envKey, key);
    ok(`provider '${args.id}' added`);
  }
}

async function mergeEnv(k: string, v: string): Promise<void> {
  const envPath = paths(resolveProfile()).env;
  await mkdir(dirname(envPath), { recursive: true, mode: 0o700 });
  let txt = '';
  try { txt = await readFile(envPath, 'utf8'); } catch {}
  const lines = txt.split('\n').filter((l) => !l.startsWith(`${k}=`));
  lines.push(`${k}=${v}`);
  await writeFile(envPath, lines.filter(Boolean).join('\n') + '\n', { mode: 0o600 });
}

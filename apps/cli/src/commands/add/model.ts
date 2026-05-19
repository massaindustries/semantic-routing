import { Command, Args, Flags } from '@oclif/core';
import { loadConfigRaw } from '../../lib/config/load.js';
import { ConfigSchema } from '../../lib/config/schema.js';
import { saveConfig } from '../../lib/config/save.js';
import { ok, err } from '../../lib/ui/banners.js';
import { catalog } from '../../lib/catalog/index.js';

export default class AddModel extends Command {
  static description = 'Add a model entry to model_config';
  static args = { id: Args.string({ required: true }) };
  static flags = {
    provider: Flags.string({ required: true, description: 'provider id (must already exist)' }),
    'param-size': Flags.string({ description: 'e.g. 9b, 70b, unknown' }),
    'reasoning-family': Flags.string(),
  };
  async run(): Promise<void> {
    const { args, flags } = await this.parse(AddModel);
    const raw = (await loadConfigRaw()) as any;
    raw.providers = raw.providers ?? {};
    if (!raw.providers[flags.provider]) { err(`provider '${flags.provider}' not configured. add it first via \`brick add provider\``); this.exit(1); }
    const catEntry = Object.values(catalog).flatMap((c) => c.models).find((m) => m.id === args.id);
    raw.model_config = raw.model_config ?? {};
    raw.model_config[args.id] = {
      preferred_endpoints: [flags.provider],
      param_size: flags['param-size'] ?? catEntry?.param_size ?? 'unknown',
      ...(flags['reasoning-family'] ?? catEntry?.reasoning_family
        ? { reasoning_family: flags['reasoning-family'] ?? catEntry?.reasoning_family }
        : {}),
    };
    const cfg = ConfigSchema.parse(raw);
    await saveConfig(cfg);
    ok(`model '${args.id}' added (provider=${flags.provider})`);
  }
}

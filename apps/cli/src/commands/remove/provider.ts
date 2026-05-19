import { Command, Args } from '@oclif/core';
import { loadConfigRaw } from '../../lib/config/load.js';
import { ConfigSchema } from '../../lib/config/schema.js';
import { saveConfig } from '../../lib/config/save.js';
import { ok, err } from '../../lib/ui/banners.js';

export default class RemoveProvider extends Command {
  static description = 'Remove a provider';
  static args = { id: Args.string({ required: true }) };
  async run(): Promise<void> {
    const { args } = await this.parse(RemoveProvider);
    const raw = (await loadConfigRaw()) as any;
    if (!raw.providers?.[args.id]) { err(`provider '${args.id}' not found`); this.exit(1); }
    delete raw.providers[args.id];
    if (raw.provider_profiles) delete raw.provider_profiles[args.id];
    if (Array.isArray(raw.provider_endpoints)) raw.provider_endpoints = raw.provider_endpoints.filter((v: any) => v.name !== args.id);
    if (raw.model_config) {
      for (const [m, mc] of Object.entries<any>(raw.model_config)) {
        mc.preferred_endpoints = (mc.preferred_endpoints ?? []).filter((p: string) => p !== args.id);
        if (mc.preferred_endpoints.length === 0) delete raw.model_config[m];
      }
    }
    const cfg = ConfigSchema.parse(raw);
    await saveConfig(cfg);
    ok(`provider '${args.id}' removed`);
  }
}

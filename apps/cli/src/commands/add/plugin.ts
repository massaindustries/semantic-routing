import { Command, Args, Flags } from '@oclif/core';
import { loadConfigRaw } from '../../lib/config/load.js';
import { ConfigSchema } from '../../lib/config/schema.js';
import { saveConfig } from '../../lib/config/save.js';
import { ok } from '../../lib/ui/banners.js';

export default class AddPlugin extends Command {
  static description = 'Enable a plugin (pii_detection|jailbreak_guard|semantic_cache|prompt_guard)';
  static args = { name: Args.string({ required: true }) };
  static flags = { action: Flags.string({ default: 'block' }) };
  async run(): Promise<void> {
    const { args, flags } = await this.parse(AddPlugin);
    const raw = (await loadConfigRaw()) as any;
    raw.plugins = raw.plugins ?? {};
    raw.plugins[args.name] = { enabled: true, action: flags.action };
    const cfg = ConfigSchema.parse(raw);
    await saveConfig(cfg);
    ok(`plugin '${args.name}' enabled (action=${flags.action})`);
  }
}

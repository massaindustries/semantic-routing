import { Command, Args } from '@oclif/core';
import { loadConfigRaw } from '../../lib/config/load.js';
import { ConfigSchema } from '../../lib/config/schema.js';
import { saveConfig } from '../../lib/config/save.js';
import { ok, err } from '../../lib/ui/banners.js';

export default class RemovePlugin extends Command {
  static description = 'Disable / remove a plugin';
  static args = { name: Args.string({ required: true }) };
  async run(): Promise<void> {
    const { args } = await this.parse(RemovePlugin);
    const raw = (await loadConfigRaw()) as any;
    if (!raw.plugins?.[args.name]) { err(`plugin '${args.name}' not configured`); this.exit(1); }
    delete raw.plugins[args.name];
    const cfg = ConfigSchema.parse(raw);
    await saveConfig(cfg);
    ok(`plugin '${args.name}' removed`);
  }
}

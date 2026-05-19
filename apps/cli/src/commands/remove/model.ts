import { Command, Args } from '@oclif/core';
import { loadConfigRaw } from '../../lib/config/load.js';
import { ConfigSchema } from '../../lib/config/schema.js';
import { saveConfig } from '../../lib/config/save.js';
import { ok, err } from '../../lib/ui/banners.js';

export default class RemoveModel extends Command {
  static description = 'Remove a model entry';
  static args = { id: Args.string({ required: true }) };
  async run(): Promise<void> {
    const { args } = await this.parse(RemoveModel);
    const raw = (await loadConfigRaw()) as any;
    if (!raw.model_config?.[args.id]) { err(`model '${args.id}' not found`); this.exit(1); }
    delete raw.model_config[args.id];
    const cfg = ConfigSchema.parse(raw);
    await saveConfig(cfg);
    ok(`model '${args.id}' removed`);
  }
}

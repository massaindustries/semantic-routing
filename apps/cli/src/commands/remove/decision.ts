import { Command, Args } from '@oclif/core';
import { loadConfigRaw } from '../../lib/config/load.js';
import { ConfigSchema } from '../../lib/config/schema.js';
import { saveConfig } from '../../lib/config/save.js';
import { ok, err } from '../../lib/ui/banners.js';

export default class RemoveDecision extends Command {
  static description = 'Remove a decision by name';
  static args = { name: Args.string({ required: true }) };
  async run(): Promise<void> {
    const { args } = await this.parse(RemoveDecision);
    const raw = (await loadConfigRaw()) as any;
    const before = (raw.decisions ?? []).length;
    raw.decisions = (raw.decisions ?? []).filter((d: any) => d.name !== args.name);
    if (raw.decisions.length === before) { err(`decision '${args.name}' not found`); this.exit(1); }
    const cfg = ConfigSchema.parse(raw);
    await saveConfig(cfg);
    ok(`decision '${args.name}' removed (${cfg.decisions.length} remaining)`);
  }
}

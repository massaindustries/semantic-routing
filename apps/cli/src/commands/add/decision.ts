import { Command } from '@oclif/core';
import { loadConfigRaw } from '../../lib/config/load.js';
import { ConfigSchema } from '../../lib/config/schema.js';
import { saveConfig } from '../../lib/config/save.js';
import { runDecisionBuilder } from '../../lib/wizard/steps/decisions.js';
import { ok } from '../../lib/ui/banners.js';

export default class AddDecision extends Command {
  static description = 'Append a routing decision via interactive builder';
  async run(): Promise<void> {
    const raw = (await loadConfigRaw()) as any;
    const available = Object.keys(raw.model_config ?? {});
    const newDecisions = await runDecisionBuilder(available);
    raw.decisions = [...(raw.decisions ?? []), ...newDecisions];
    const cfg = ConfigSchema.parse(raw);
    await saveConfig(cfg);
    ok(`added ${newDecisions.length} decision(s); total ${cfg.decisions.length}`);
  }
}

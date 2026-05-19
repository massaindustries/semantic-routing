import { Args, Command } from '@oclif/core';
import { profileExists, updateState } from '../../lib/config/paths.js';
import { err, ok } from '../../lib/ui/banners.js';

export default class ConfigUse extends Command {
  static description = 'Set the active profile (used as default by serve/chat/route/generate)';
  static args = {
    profile: Args.string({ required: true, description: 'profile name to mark as active' }),
  };

  async run(): Promise<void> {
    const { args } = await this.parse(ConfigUse);
    if (!profileExists(args.profile)) { err(`profile '${args.profile}' not found`); this.exit(1); }
    updateState({ activeProfile: args.profile });
    ok(`active profile → ${args.profile}`);
  }
}

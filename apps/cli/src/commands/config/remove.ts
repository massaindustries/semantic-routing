import { Args, Command, Flags } from '@oclif/core';
import * as p from '@clack/prompts';
import { rm } from 'node:fs/promises';
import { paths, profileExists, readState, updateState, listProfiles } from '../../lib/config/paths.js';
import { err, ok, warn } from '../../lib/ui/banners.js';

export default class ConfigRemove extends Command {
  static description = 'Delete a profile (config + docker-compose + .env)';
  static aliases = ['config:rm', 'config:delete'];
  static args = {
    profile: Args.string({ required: true, description: 'profile name to remove' }),
  };
  static flags = {
    force: Flags.boolean({ char: 'f', description: 'skip confirmation' }),
  };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(ConfigRemove);
    if (!profileExists(args.profile)) { err(`profile '${args.profile}' not found`); this.exit(1); }

    const state = readState();
    if (state.runningProfile === args.profile) {
      err(`profile '${args.profile}' is running. run \`brick down ${args.profile}\` first.`);
      this.exit(1);
    }

    if (!flags.force) {
      const sure = await p.confirm({ message: `Delete profile '${args.profile}' and all its files?`, initialValue: false });
      if (p.isCancel(sure) || !sure) { warn('aborted'); return; }
    }

    await rm(paths(args.profile).profileDir, { recursive: true, force: true });

    let nextActive: string | null | undefined = undefined;
    if (state.activeProfile === args.profile) {
      const remaining = listProfiles();
      nextActive = remaining[0] ?? null;
    }
    if (nextActive !== undefined) updateState({ activeProfile: nextActive });
    ok(`removed profile '${args.profile}'${nextActive !== undefined ? ` (active → ${nextActive ?? 'none'})` : ''}`);
  }
}

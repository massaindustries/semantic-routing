import { Args, Command } from '@oclif/core';
import { dockerCompose } from '../lib/docker/run.js';
import { resolveProfile, readState, updateState } from '../lib/config/paths.js';
import { err, ok } from '../lib/ui/banners.js';

export default class Down extends Command {
  static description = 'Tear down a profile’s router container (removes container, mounts persist)';
  static args = {
    profile: Args.string({ required: false, description: 'profile name (defaults to running or active profile)' }),
  };
  async run(): Promise<void> {
    const { args } = await this.parse(Down);
    let profile: string;
    try {
      profile = resolveProfile(args.profile ?? readState().runningProfile ?? undefined);
    } catch (e: any) { err(e?.message ?? String(e)); this.exit(1); }

    const r = await dockerCompose(profile, ['down']);
    if (r.exitCode !== 0) { err(r.stderr.slice(0, 500)); this.exit(1); }

    const state = readState();
    if (state.runningProfile === profile) updateState({ runningProfile: null });
    ok(`down (profile: ${profile})`);
  }
}

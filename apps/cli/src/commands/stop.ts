import { Args, Command } from '@oclif/core';
import { dockerCompose } from '../lib/docker/run.js';
import { resolveProfile, readState, updateState } from '../lib/config/paths.js';
import { err, ok } from '../lib/ui/banners.js';

export default class Stop extends Command {
  static description = 'Stop a profile’s router container (keeps the container, data persists)';
  static args = {
    profile: Args.string({ required: false, description: 'profile name (defaults to running or active profile)' }),
  };
  async run(): Promise<void> {
    const { args } = await this.parse(Stop);
    let profile: string;
    try {
      profile = resolveProfile(args.profile ?? readState().runningProfile ?? undefined);
    } catch (e: any) { err(e?.message ?? String(e)); this.exit(1); }

    const r = await dockerCompose(profile, ['stop']);
    if (r.exitCode !== 0) { err(r.stderr.slice(0, 500)); this.exit(1); }

    const state = readState();
    if (state.runningProfile === profile) updateState({ runningProfile: null });
    ok(`stopped (profile: ${profile})`);
  }
}

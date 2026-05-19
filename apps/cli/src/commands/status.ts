import { Args, Command } from '@oclif/core';
import { dockerCompose } from '../lib/docker/run.js';
import { loadConfig } from '../lib/config/load.js';
import { resolveProfile, readState, listProfiles } from '../lib/config/paths.js';
import { info, ok, warn, err, print } from '../lib/ui/banners.js';

export default class Status extends Command {
  static description = 'Show profiles, container state, and health';
  static args = {
    profile: Args.string({ required: false, description: 'profile name (defaults to active profile)' }),
  };
  async run(): Promise<void> {
    const { args } = await this.parse(Status);
    const state = readState();
    const profs = listProfiles();
    info(`profiles: ${profs.length === 0 ? '(none)' : profs.map((p) => `${p}${state.activeProfile === p ? ' [active]' : ''}${state.runningProfile === p ? ' [running]' : ''}`).join('  ·  ')}`);

    let profile: string;
    try { profile = resolveProfile(args.profile); }
    catch (e: any) { err(e?.message ?? String(e)); return; }

    const ps = await dockerCompose(profile, ['ps']);
    info(`container state (profile: ${profile}):`);
    print(ps.stdout || '(no compose project up)');
    try {
      const cfg = await loadConfig(profile);
      const r = await fetch(`http://localhost:${cfg.server_port}/health`, { signal: AbortSignal.timeout(3000) });
      if (r.ok) ok(`/health → ${r.status}`);
      else warn(`/health → ${r.status}`);
    } catch (e: any) { err(`health probe failed: ${e?.message ?? e}`); }
  }
}

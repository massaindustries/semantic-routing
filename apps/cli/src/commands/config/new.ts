import { Args, Command, Flags } from '@oclif/core';
import * as p from '@clack/prompts';
import { runWizard } from '../../lib/wizard/run.js';
import { dockerInstalled } from '../../lib/docker/run.js';
import { banner, err, info, ok } from '../../lib/ui/banners.js';
import { paths, listProfiles, readState, updateState, profileExists } from '../../lib/config/paths.js';
import { stat } from 'node:fs/promises';

// Docker Compose project names must match [a-z0-9_-]+; keep this in sync with
// the wizard validator in src/commands/init.ts so a created profile can be served.
const NAME_RE = /^[a-z0-9_-]{1,64}$/;

export default class ConfigNew extends Command {
  static description = 'Create a new profile via the guided wizard';
  static args = {
    profile: Args.string({ required: false, description: 'profile name (prompted if omitted)' }),
  };
  static flags = {
    force: Flags.boolean({ char: 'f', description: 'overwrite if profile already exists' }),
  };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(ConfigNew);
    banner();

    let profile = args.profile;
    if (!profile) {
      const name = await p.text({
        message: 'Profile name:',
        placeholder: 'default',
        validate: (s) => NAME_RE.test(s.trim()) ? undefined : 'use lowercase letters, digits, - or _, 1–64 chars',
      });
      if (p.isCancel(name)) { p.cancel('aborted'); process.exit(0); }
      profile = String(name).trim();
    }

    if (!NAME_RE.test(profile)) { err(`invalid profile name '${profile}'`); this.exit(1); }
    if (profileExists(profile) && !flags.force) {
      err(`profile '${profile}' already exists. use --force to overwrite or pick another name.`);
      this.exit(1);
    }
    if (!(await dockerInstalled())) { err('docker not found in PATH. install Docker first.'); this.exit(1); }
    else ok('docker available');

    info(`creating profile: ${profile}`);
    try { await stat(paths(profile).config); } catch {}
    await runWizard(profile);

    const state = readState();
    if (!state.activeProfile || listProfiles().length === 1) {
      updateState({ activeProfile: profile });
      ok(`profile '${profile}' set as active`);
    } else {
      info(`run \`brick config use ${profile}\` to make this the active profile.`);
    }
  }
}

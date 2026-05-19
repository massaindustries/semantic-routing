import { Args, Command, Flags } from '@oclif/core';
import * as p from '@clack/prompts';
import { runWizard } from '../lib/wizard/run.js';
import { dockerInstalled } from '../lib/docker/run.js';
import { banner, err, info, ok, warn } from '../lib/ui/banners.js';
import { paths, listProfiles, readState, updateState } from '../lib/config/paths.js';
import { stat } from 'node:fs/promises';

export default class Init extends Command {
  static description = 'Run guided wizard and create a new profile (or overwrite an existing one)';
  static args = {
    profile: Args.string({ required: false, description: 'profile name (defaults to "default")' }),
  };
  static flags = {
    force: Flags.boolean({ char: 'f', description: 'overwrite existing profile config without confirmation' }),
  };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(Init);
    banner();
    let profile = args.profile;
    if (profile && !/^[a-z0-9_-]+$/.test(profile)) {
      err(`invalid profile name '${profile}'. use only lowercase letters, digits, - or _`);
      this.exit(1);
    }
    if (!profile) {
      const nameRaw = await p.text({
        message: 'Profile name:',
        placeholder: 'default',
        defaultValue: 'default',
        validate: (v) => {
          const s = v.trim();
          if (!s) return 'Profile name cannot be empty';
          if (!/^[a-z0-9_-]+$/.test(s)) return 'Use only lowercase letters, digits, - or _';
        },
      });
      if (p.isCancel(nameRaw)) { p.cancel('aborted'); this.exit(0); }
      profile = (nameRaw as string).trim();
    }
    if (!(await dockerInstalled())) {
      err('docker not found in PATH. install Docker first.');
      this.exit(1);
    } else {
      ok('docker available');
    }
    const pp = paths(profile);
    try {
      const s = await stat(pp.config);
      if (s.isFile() && !flags.force) {
        warn(`profile '${profile}' already exists at ${pp.config}. running wizard will overwrite on confirm.`);
      }
    } catch {}
    info(`profile: ${profile}`);
    await runWizard(profile);

    const state = readState();
    if (!state.activeProfile || listProfiles().length === 1) {
      updateState({ activeProfile: profile });
      ok(`profile '${profile}' set as active`);
    }
  }
}

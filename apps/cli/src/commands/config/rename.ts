import { Args, Command } from '@oclif/core';
import { rename } from 'node:fs/promises';
import { paths, profileExists, readState, updateState } from '../../lib/config/paths.js';
import { writeCompose } from '../../lib/docker/compose.js';
import { loadConfig } from '../../lib/config/load.js';
import { err, ok } from '../../lib/ui/banners.js';

// Docker Compose project names must match [a-z0-9_-]+; keep in sync with new.ts + init.ts.
const NAME_RE = /^[a-z0-9_-]{1,64}$/;

export default class ConfigRename extends Command {
  static description = 'Rename a profile';
  static args = {
    old: Args.string({ required: true, description: 'current profile name' }),
    new: Args.string({ required: true, description: 'new profile name' }),
  };

  async run(): Promise<void> {
    const { args } = await this.parse(ConfigRename);
    if (!NAME_RE.test(args.new)) { err(`invalid new profile name '${args.new}'`); this.exit(1); }
    if (!profileExists(args.old)) { err(`profile '${args.old}' not found`); this.exit(1); }
    if (profileExists(args.new)) { err(`profile '${args.new}' already exists`); this.exit(1); }

    const state = readState();
    if (state.runningProfile === args.old) {
      err(`profile '${args.old}' is running. run \`brick stop ${args.old}\` first.`);
      this.exit(1);
    }

    const oldPaths = paths(args.old);
    const newPaths = paths(args.new);
    await rename(oldPaths.profileDir, newPaths.profileDir);

    // regenerate docker-compose to update project_name + mount paths.
    try {
      const cfg = await loadConfig(args.new);
      await writeCompose({ profile: args.new, port: cfg.server_port });
    } catch {
      // best-effort; user can re-run config edit if needed.
    }

    if (state.activeProfile === args.old) updateState({ activeProfile: args.new });
    ok(`renamed '${args.old}' → '${args.new}'`);
  }
}

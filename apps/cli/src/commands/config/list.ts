import { Command, Flags } from '@oclif/core';
import chalk from 'chalk';
import { listProfiles, paths, readState } from '../../lib/config/paths.js';
import { loadConfig } from '../../lib/config/load.js';
import { makeTable, renderTable } from '../../lib/ui/tables.js';
import { info, print } from '../../lib/ui/banners.js';

export default class ConfigList extends Command {
  static description = 'List all profiles';
  static aliases = ['profiles'];
  static flags = {
    json: Flags.boolean({ default: false, description: 'output JSON' }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(ConfigList);
    const profs = listProfiles();
    const state = readState();

    if (flags.json) {
      const rows = await Promise.all(profs.map(async (name) => {
        try {
          const cfg = await loadConfig(name);
          return {
            name,
            active: state.activeProfile === name,
            running: state.runningProfile === name,
            default_model: cfg.default_model,
            providers: Object.keys(cfg.providers ?? {}),
            port: cfg.server_port,
          };
        } catch {
          return { name, active: state.activeProfile === name, running: state.runningProfile === name, error: 'invalid config' };
        }
      }));
      console.log(JSON.stringify(rows, null, 2));
      return;
    }

    if (profs.length === 0) {
      info('no profiles configured. run `brick config new <name>` to create one.');
      return;
    }

    const t = makeTable(['', 'profile', 'default model', 'providers', 'port', 'state']);
    for (const name of profs) {
      let model = '-'; let providers = '-'; let port = '-';
      try {
        const cfg = await loadConfig(name);
        model = cfg.default_model;
        providers = Object.keys(cfg.providers ?? {}).join(',');
        port = String(cfg.server_port);
      } catch { model = chalk.red('(invalid)'); }
      const flagsCol: string[] = [];
      if (state.runningProfile === name) flagsCol.push(chalk.green('● running'));
      if (state.activeProfile === name) flagsCol.push(chalk.cyan('active'));
      const marker = state.runningProfile === name ? chalk.green('●') : (state.activeProfile === name ? chalk.cyan('*') : ' ');
      t.push([marker, name, model, providers, port, flagsCol.join(' ')]);
    }
    print();
    print(renderTable(t));
    print();
    info(`config dir: ${paths(profs[0]).root}/profiles`);
  }
}

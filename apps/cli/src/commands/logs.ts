import { Args, Command, Flags } from '@oclif/core';
import { execa } from 'execa';
import { paths, resolveProfile } from '../lib/config/paths.js';

export default class Logs extends Command {
  static description = 'Stream container logs for a profile';
  static args = {
    profile: Args.string({ required: false, description: 'profile name (defaults to active profile)' }),
  };
  static flags = {
    tail: Flags.integer({ default: 100, description: 'number of lines to tail' }),
    follow: Flags.boolean({ char: 'f', default: false, description: 'follow logs' }),
  };
  async run(): Promise<void> {
    const { args, flags } = await this.parse(Logs);
    const profile = resolveProfile(args.profile);
    const pp = paths(profile);
    const dargs = ['compose', '-p', `brick-${profile}`, '-f', pp.compose, 'logs', '--tail', String(flags.tail)];
    if (flags.follow) dargs.push('-f');
    await execa('docker', dargs, { stdio: 'inherit', reject: false });
  }
}

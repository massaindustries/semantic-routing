import { execa } from 'execa';
import { paths, resolveProfile } from '../config/paths.js';

export interface ExecResult { stdout: string; stderr: string; exitCode: number }

export async function dockerCompose(profile: string | undefined, args: string[]): Promise<ExecResult> {
  const p = paths(resolveProfile(profile));
  const r = await execa('docker', ['compose', '-p', `brick-${p.profile}`, '-f', p.compose, ...args], { reject: false });
  return { stdout: r.stdout, stderr: r.stderr, exitCode: r.exitCode ?? 1 };
}

export async function dockerCmd(args: string[]): Promise<ExecResult> {
  const r = await execa('docker', args, { reject: false });
  return { stdout: r.stdout, stderr: r.stderr, exitCode: r.exitCode ?? 1 };
}

export async function dockerInstalled(): Promise<boolean> {
  try {
    const r = await execa('docker', ['--version'], { reject: false });
    return r.exitCode === 0;
  } catch {
    return false;
  }
}

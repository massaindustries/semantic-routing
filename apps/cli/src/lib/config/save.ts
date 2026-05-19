import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import yaml from 'js-yaml';
import { paths, resolveProfile } from './paths.js';
import type { BrickConfig } from './schema.js';

export async function saveConfig(cfg: BrickConfig, profile?: string): Promise<string> {
  const target = paths(profile ?? resolveProfile()).config;
  await mkdir(dirname(target), { recursive: true, mode: 0o700 });
  const dump = yaml.dump(cfg, { lineWidth: 120, noRefs: true, sortKeys: false });
  await writeFile(target, dump, { mode: 0o600 });
  return target;
}

export async function saveConfigText(content: string, profile?: string): Promise<string> {
  const target = paths(profile ?? resolveProfile()).config;
  await mkdir(dirname(target), { recursive: true, mode: 0o700 });
  await writeFile(target, content, { mode: 0o600 });
  return target;
}

export async function saveText(content: string, path: string, mode = 0o644): Promise<void> {
  await mkdir(dirname(path), { recursive: true, mode: 0o700 });
  await writeFile(path, content, { mode });
}

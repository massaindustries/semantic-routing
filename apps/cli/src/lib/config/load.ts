import { readFile } from 'node:fs/promises';
import yaml from 'js-yaml';
import { ConfigSchema, type BrickConfig } from './schema.js';
import { paths, resolveProfile } from './paths.js';

function configPath(profile?: string): string {
  return paths(resolveProfile(profile)).config;
}

export async function loadConfig(profile?: string): Promise<BrickConfig> {
  const raw = await readFile(configPath(profile), 'utf8');
  const parsed = yaml.load(raw);
  return ConfigSchema.parse(parsed);
}

export async function loadConfigRaw(profile?: string): Promise<unknown> {
  const raw = await readFile(configPath(profile), 'utf8');
  return yaml.load(raw);
}

export async function loadConfigText(profile?: string): Promise<string> {
  return readFile(configPath(profile), 'utf8');
}

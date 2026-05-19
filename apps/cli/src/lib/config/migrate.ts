import { stat, mkdir, rename } from 'node:fs/promises';
import { LEGACY, paths, profilesDir, readState, writeState, listProfiles } from './paths.js';
import { writeCompose } from '../docker/compose.js';
import { loadConfig } from './load.js';

let migrated = false;

async function exists(p: string): Promise<boolean> {
  try { await stat(p); return true; } catch { return false; }
}

/**
 * Migrate a legacy `~/.brick/{config.yaml,docker-compose.yml,.env,models/}`
 * layout into `~/.brick/profiles/default/`. Idempotent: runs at most once
 * per process and is a no-op once `profiles/` already exists.
 */
export async function migrateLegacyLayout(): Promise<{ migrated: boolean; reason?: string }> {
  if (migrated) return { migrated: false, reason: 'already-migrated' };
  migrated = true;

  const hasProfiles = await exists(profilesDir());
  const hasLegacyConfig = await exists(LEGACY.config);
  if (hasProfiles || !hasLegacyConfig) return { migrated: false };

  const target = paths('default');
  await mkdir(target.profileDir, { recursive: true, mode: 0o700 });

  for (const [src, dst] of [
    [LEGACY.config, target.config],
    [LEGACY.env, target.env],
    [LEGACY.models, target.models],
  ] as const) {
    if (await exists(src)) {
      try { await rename(src, dst); } catch { /* ignore individual move errors */ }
    }
  }

  // Compose was rendered with legacy ~/.brick/config.yaml + ~/.brick/.env paths.
  // Regenerate from the new profile layout so bind/env_file paths point at the
  // moved files; fall back to copying the legacy compose only if regeneration fails.
  try {
    const cfg = await loadConfig('default');
    await writeCompose({ profile: 'default', port: cfg.server_port });
  } catch {
    if (await exists(LEGACY.compose)) {
      try { await rename(LEGACY.compose, target.compose); } catch { /* best-effort */ }
    }
  }

  const state = readState();
  if (!state.activeProfile) {
    state.activeProfile = 'default';
    writeState(state);
  }

  return { migrated: true };
}

/** Run migration synchronously-style (fire-and-forget) for hooks. */
export async function ensureMigrated(): Promise<void> {
  await migrateLegacyLayout();
  // Self-heal: if state has no active profile but at least one profile exists, pick the first.
  const state = readState();
  if (!state.activeProfile) {
    const profs = listProfiles();
    if (profs.length > 0) writeState({ ...state, activeProfile: profs[0] });
  }
}

import type { Hook } from '@oclif/core';
import { printLogo } from '../lib/ui/banners.js';
import { ensureMigrated } from '../lib/config/migrate.js';

const HELP_TRIGGERS = new Set(['help', '--help', '-h']);
// Commands that already render their own banner internally — skip to avoid double logo.
const COMMANDS_WITH_OWN_BANNER = new Set(['init', 'serve']);

const hook: Hook<'init'> = async function (_opts) {
  // Idempotent legacy → multi-profile migration. Runs once per process.
  try { await ensureMigrated(); } catch { /* never block CLI startup on migration */ }

  // opts.argv contains only the *command's* args (after the command name has been resolved),
  // so we read the original process.argv to know which command was invoked.
  const argv = process.argv.slice(2);
  const first = argv[0];

  // No args at all → oclif will show help. Print logo first.
  if (argv.length === 0) {
    printLogo();
    return;
  }

  // Explicit help: `brick help`, `brick --help`, `brick -h`.
  if (first && HELP_TRIGGERS.has(first)) {
    printLogo();
    return;
  }

  // init/serve already print logo themselves via banner() — skip to avoid double logo.
  if (first && COMMANDS_WITH_OWN_BANNER.has(first)) return;

  // Any other command (chat, route, generate, status, logs, stop, add, remove): no logo.
};

export default hook;

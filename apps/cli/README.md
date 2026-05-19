# `apps/cli/`: `brick` CLI (`@regolo-ai/brick`)

TypeScript / oclif / ink companion CLI to self-host the Brick router with one command. Will be published on npm as `@regolo-ai/brick`.

## Install

The npm package is not yet published (see the root [Distribution roadmap](../../README.md#distribution-roadmap)). Install from source:

```bash
git clone https://github.com/regolo-ai/brick-SR1.git
cd brick-SR1/apps/cli
npm install
npm run build
npm link        # makes `brick` available on $PATH
```

Requires Node >= 18 and Docker.

## Commands

| Command | Purpose |
|---------|---------|
| `brick init [profile]` | Guided wizard → creates `~/.brick/profiles/<name>/{config.yaml, docker-compose.yml, .env}` |
| `brick serve [profile]` | `docker compose up -d` for the active profile (pulls `ghcr.io/regolo-ai/brick:latest` if missing) |
| `brick chat` | TUI chat (ink: bottom input + scrolling history, Claude Code-style) |
| `brick generate "<prompt>"` | One-shot completion (stdout) |
| `brick route "<prompt>"` | Show routing decision (selected backend + latency) without running generation if `--no-generate` |
| `brick status` | Active profile + container state |
| `brick logs` | Tail container logs |
| `brick stop` / `brick down` | Stop / down docker compose |
| `brick config new\|use\|edit\|list\|remove [profile]` | Manage YAML profiles |
| `brick add\|remove provider\|model\|decision\|plugin` | Edit current profile interactively |
| `brick claude status` | Show wiring + recent stats for Anthropic passthrough (Claude Code integration) |

Common flags: `--profile <name>`, `--thinking off\|low\|med\|high\|auto`, `--json`.

## Configuration

```
~/.brick/
├── state.json                  # {activeProfile, runningProfile}
└── profiles/
    └── <name>/
        ├── config.yaml         # router config (schema mirrors apps/router/config/config.yaml)
        ├── docker-compose.yml  # rendered from templates/docker-compose.yaml.hbs
        ├── .env                # API keys (REGOLO_API_KEY, OPENROUTER_API_KEY, …) chmod 600
        └── models/             # optional Docker volume mount for cached HF models
```

Environment overrides:
- `BRICK_HOME`: alternate root (default `~/.brick`).
- `BRICK_PROFILE`: alternate active profile (default = `state.json`'s `activeProfile`).
- `BRICK_IMAGE`: alternate Docker image (default `ghcr.io/regolo-ai/brick:latest`).
- `REGOLO_API_KEY`: provider key (read by the router via `Authorization: Bearer ...`).
- `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`: used by Claude Code passthrough.

## Build & test

```bash
cd apps/cli
npm install
npm run build         # tsc -b → dist/
npm run lint          # tsc --noEmit
npm test              # vitest
npm run test:random   # custom random-session harness (test/random.ts)
```

After build, the CLI is invocable as `./bin/run.js`.

## Publishing

The package is scoped (`@regolo-ai/brick`) and published with `--access public`. CI handles publishing on tag `v*` (see `.github/workflows/npm-publish.yml`).

Local dry-run:

```bash
npm pack
# inspect the tarball before npm publish
```

## Source layout

```
apps/cli/
├── bin/
│   ├── run.js                  # oclif entry (production)
│   └── dev.js                  # oclif entry (ts-node loader)
├── src/
│   ├── commands/               # oclif commands (chat, serve, route, init, …)
│   │   ├── add/, remove/       # topic groups
│   │   ├── config/             # profile management
│   │   └── claude/             # Anthropic passthrough wiring inspector
│   ├── lib/
│   │   ├── client/             # OpenAI-compatible HTTP client (SSE streaming)
│   │   ├── chat-tui/           # ink components (App, Welcome, BABL pane, SlashPopup, …)
│   │   ├── config/             # paths, load, validate (zod schema), migrate
│   │   ├── config-ai/          # interactive `config ai` agent (React/ink)
│   │   ├── docker/             # image / compose / run helpers
│   │   ├── ui/                 # banners, colors
│   │   └── wizard/             # guided init prompts
│   └── hooks/
│       └── init.ts             # oclif lifecycle hook (legacy migration, banner)
├── templates/
│   └── docker-compose.yaml.hbs # Handlebars template for `brick init`
├── test/
│   ├── random.ts               # random-session test harness
│   └── fixtures/prompts.json
├── package.json
└── tsconfig.json
```

## Related

- Router architecture and config knobs: [apps/router/README.md](../router/README.md).
- One-line Docker quickstart (no CLI): [docs/quickstart/quick.md](../../docs/quickstart/quick.md).
- Full CLI walkthrough: [docs/quickstart/serve.md](../../docs/quickstart/serve.md).

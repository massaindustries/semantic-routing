import yaml from 'js-yaml';
import { readFile, writeFile } from 'node:fs/promises';
import { ConfigSchema } from '../config/schema.js';
import { paths } from '../config/paths.js';
import type { ToolDef } from '../client/openai.js';

export const CONFIG_TOOLS: ToolDef[] = [
  {
    type: 'function',
    function: {
      name: 'read_config',
      description: 'Read the current config.yaml of the profile being edited. Returns the full YAML as a string.',
      parameters: { type: 'object', properties: {}, additionalProperties: false },
    },
  },
  {
    type: 'function',
    function: {
      name: 'validate_config',
      description: 'Validate a candidate YAML against the schema without writing. Returns { valid, errors? }.',
      parameters: {
        type: 'object',
        properties: {
          yaml_text: { type: 'string', description: 'Full YAML content to validate.' },
        },
        required: ['yaml_text'],
        additionalProperties: false,
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'propose_patch',
      description: 'Propose a new full YAML for the config. The CLI will show a diff to the user and ask for confirmation. Returns { applied, errors? }.',
      parameters: {
        type: 'object',
        properties: {
          rationale: { type: 'string', description: 'One-paragraph explanation of why this change is needed.' },
          yaml_text: { type: 'string', description: 'Full new YAML (not a diff).' },
        },
        required: ['rationale', 'yaml_text'],
        additionalProperties: false,
      },
    },
  },
];

export interface ToolContext {
  profile: string;
  /** Returns true if the user accepted the patch, false if rejected. */
  confirmPatch: (rationale: string, oldYaml: string, newYaml: string) => Promise<boolean>;
}

export interface ToolResult {
  ok: boolean;
  data?: any;
  error?: string;
}

export async function dispatchTool(name: string, argsJson: string, ctx: ToolContext): Promise<ToolResult> {
  let args: any;
  try { args = argsJson ? JSON.parse(argsJson) : {}; }
  catch (e: any) { return { ok: false, error: `invalid JSON arguments: ${e?.message ?? e}` }; }

  if (name === 'read_config') return readConfigTool(ctx.profile);
  if (name === 'validate_config') return validateConfigTool(args.yaml_text);
  if (name === 'propose_patch') return proposePatchTool(ctx, args.rationale, args.yaml_text);
  return { ok: false, error: `unknown tool: ${name}` };
}

async function readConfigTool(profile: string): Promise<ToolResult> {
  try {
    const txt = await readFile(paths(profile).config, 'utf8');
    return { ok: true, data: { yaml_text: txt } };
  } catch (e: any) {
    return { ok: false, error: `read failed: ${e?.message ?? e}` };
  }
}

function validateYaml(yamlText: string): { valid: boolean; errors?: string[] } {
  let parsed: any;
  try { parsed = yaml.load(yamlText); }
  catch (e: any) { return { valid: false, errors: [`YAML parse error: ${e?.message ?? e}`] }; }
  const r = ConfigSchema.safeParse(parsed);
  if (r.success) return { valid: true };
  const issues = r.error.issues.map((i) => `${i.path.join('.') || '<root>'}: ${i.message}`);
  return { valid: false, errors: issues };
}

function validateConfigTool(yamlText: string): ToolResult {
  if (!yamlText) return { ok: false, error: 'yaml_text is required' };
  return { ok: true, data: validateYaml(yamlText) };
}

async function proposePatchTool(ctx: ToolContext, rationale: string, yamlText: string): Promise<ToolResult> {
  if (!yamlText) return { ok: false, error: 'yaml_text is required' };
  const validation = validateYaml(yamlText);
  if (!validation.valid) return { ok: true, data: { applied: false, errors: validation.errors } };

  const target = paths(ctx.profile).config;
  let oldYaml = '';
  try { oldYaml = await readFile(target, 'utf8'); } catch {}

  const accepted = await ctx.confirmPatch(rationale ?? '', oldYaml, yamlText);
  if (!accepted) return { ok: true, data: { applied: false, rejected_by_user: true } };

  try {
    await writeFile(target, yamlText, { mode: 0o600 });
    return { ok: true, data: { applied: true, path: target } };
  } catch (e: any) {
    return { ok: false, error: `write failed: ${e?.message ?? e}` };
  }
}

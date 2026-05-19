import type { BrickConfig } from './schema.js';

export function extractUniqueModels(cfg: BrickConfig): string[] {
  const seen = new Set<string>();
  for (const model of cfg.skill_router?.models ?? []) {
    if (model.model && model.model.trim()) seen.add(model.model.trim());
  }
  for (const decision of cfg.decisions) {
    for (const ref of decision.modelRefs) {
      if (ref.model && ref.model.trim()) seen.add(ref.model.trim());
    }
  }
  return Array.from(seen).sort();
}

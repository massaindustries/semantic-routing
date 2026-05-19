import type { CatalogProvider } from './index.js';

export const localCatalog: CatalogProvider = {
  id: 'local',
  label: 'Local OpenAI-compatible',
  type: 'openai_compatible',
  base_url: 'http://host.docker.internal:11434/v1',
  env_key: 'LOCAL_API_KEY',
  models: [],
  multimodal: {},
};

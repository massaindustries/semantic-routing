import type { CatalogProvider } from './index.js';

export const regoloCatalog: CatalogProvider = {
  id: 'regolo',
  label: 'Regolo AI',
  type: 'openai_compatible',
  base_url: 'https://api.regolo.ai/v1',
  env_key: 'REGOLO_API_KEY',
  models: [
    { id: 'qwen3.5-9b', label: 'Qwen 3.5 9B', param_size: '9b', reasoning_family: 'qwen3' },
    { id: 'deepseek-v4-flash', label: 'DeepSeek V4 Flash', param_size: 'unknown' },
    { id: 'kimi2.6', label: 'Kimi 2.6', param_size: 'unknown', reasoning_family: 'qwen3' },
    { id: 'qwen3.5-122b', label: 'Qwen 3.5 122B', param_size: '122b', reasoning_family: 'qwen3' },
    { id: 'qwen3-coder-next', label: 'Qwen3 Coder Next', param_size: '80b' },
    { id: 'minimax-m2.5', label: 'MiniMax M2.5', param_size: '400b', reasoning_family: 'minimax' },
    { id: 'gpt-oss-120b', label: 'GPT-OSS 120B', param_size: '120b' },
    { id: 'gpt-oss-20b', label: 'GPT-OSS 20B', param_size: '20b' },
    { id: 'Llama-3.3-70B-Instruct', label: 'Llama 3.3 70B', param_size: '70b' },
    { id: 'Llama-3.1-8B-Instruct', label: 'Llama 3.1 8B', param_size: '8b' },
    { id: 'gemma4-31b', label: 'Gemma 4 31B', param_size: '31b' },
    { id: 'mistral-small3.2', label: 'Mistral Small 3.2', param_size: '24b' },
    { id: 'apertus-70b', label: 'Apertus 70B', param_size: '70b' },
  ],
  multimodal: {
    stt: { model: 'faster-whisper-large-v3', endpoint: 'https://api.regolo.ai/v1/audio/transcriptions' },
    ocr: { model: 'deepseek-ocr-2', endpoint: 'https://api.regolo.ai/v1/chat/completions' },
    vision: { model: 'qwen3.5-122b', endpoint: 'https://api.regolo.ai/v1/chat/completions' },
  },
};

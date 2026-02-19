import asyncio
from my_model.gateway import app, set_workspace_config
from my_model.config import WorkspaceConfig, ProviderConfig, ModelConfig
from fastapi.testclient import TestClient

# Create a minimal config with mock provider
provider = ProviderConfig(provider_id='mock', base_url='http://example.com')
model = ModelConfig(backend_id='mock-test', provider_id='mock', model_id='dummy')
workspace = WorkspaceConfig(alias='test', providers=[provider], models=[model])
set_workspace_config(workspace)

client = TestClient(app)

# Test health endpoint
resp = client.get('/health')
print('Health status:', resp.status_code, resp.json())

# Test models endpoint
resp = client.get('/v1/models')
print('Models response:', resp.status_code, resp.json())

# Test chat completions non-stream with correct alias (uses mock provider)
payload = {'model': 'test', 'messages': [{'role': 'user', 'content': 'Hello'}], 'stream': False}
resp = client.post('/v1/chat/completions', json=payload)
print('Chat completions status:', resp.status_code, resp.json())

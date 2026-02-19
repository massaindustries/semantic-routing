import json
from fastapi.testclient import TestClient

from my_model.gateway import app, set_workspace_config
from my_model.config import WorkspaceConfig, ProviderConfig, ModelConfig

# Setup minimal workspace config with MockProvider
provider = ProviderConfig(provider_id='mock', base_url='http://example.com')
model = ModelConfig(backend_id='mock-test', provider_id='mock', model_id='dummy')
workspace = WorkspaceConfig(alias='test', providers=[provider], models=[model])
set_workspace_config(workspace)

client = TestClient(app)

def test_chat_completions_streaming():
    payload = {
        "model": "test",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }
    # Use stream=True to get a streaming response
    response = client.post('/v1/chat/completions', json=payload, stream=True)
    # Basic checks
    assert response.status_code == 200
    # Header may contain charset; check startswith
    assert response.headers.get('content-type', '').startswith('text/event-stream')

    # Collect streamed chunks (simulate early disconnect)
    chunks = []
    for chunk in response.iter_content(chunk_size=1024):
        chunks.append(chunk)
        # Break early to simulate client disconnect
        break
    # Ensure at least one SSE data chunk was received
    assert any(b'data:' in c for c in chunks)

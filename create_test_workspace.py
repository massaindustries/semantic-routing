import os
from my_model.config import WorkspaceConfig, ProviderConfig, ModelConfig

alias = "testmodel"
# Remove any existing workspace dir for clean install
workspace_dir = os.path.expanduser(f"~/.my-model/{alias}")
if os.path.isdir(workspace_dir):
    import shutil
    shutil.rmtree(workspace_dir)

# Create workspace config with mock provider
ws = WorkspaceConfig(alias=alias)
# Add mock provider (base_url is dummy, not used for mock)
ws.providers.append(ProviderConfig(provider_id="mock", base_url="http://mock", api_key=None))
# Add a model using mock provider
ws.models.append(ModelConfig(backend_id="mock-dummy", provider_id="mock", model_id="dummy", tags=[]))
ws.save()
print(f"Workspace {alias} created at {ws._workspace_dir}")

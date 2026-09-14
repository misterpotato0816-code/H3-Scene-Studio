import copy
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock
from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server
from h3app import config

tmp = tempfile.TemporaryDirectory(prefix="h3-ui-")
root = Path(tmp.name)

class PreviewConfig(config.Config):
    @property
    def projects_dir(self): return root / "projects"
    @property
    def story_dir(self): return root / "stories"
    @property
    def debug_dir(self): return root / "debug"
    @property
    def app_dir(self): return root

data = copy.deepcopy(config.DEFAULT_CONFIG)
data.update(comfy_dir=str(root / "comfy"), media_root=str(root / "media"), auto_launch_comfy=False)
server.presets_mod.STORE_PATH = root / "actions.json"
app = server.create_app(PreviewConfig(data))
app["client"].is_reachable = AsyncMock(return_value=False)
app.on_startup.clear()
web.run_app(app, host="127.0.0.1", port=8799, print=None)


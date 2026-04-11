import json

class ArtifactManifest:
    def __init__(self, run_dir, run_id):
        self.run_id = run_id
        self.run_dir = run_dir
        self.manifest = None

    def build_manifest(self):
        self.manifest = {
            "run_id": self.run_id,
            "artifacts": {
                "model": {"path": "models/model.pkl", "upload": True},
                "metadata": {"path": "metadata.json", "upload": True},
                "config": {"path": "config.json", "upload": True},
                "metrics": {"path": "metrics/", "upload": False},
                "analysis": {"path": "analysis/", "upload": False},
                "backtest": {"path": "backtest/", "upload": False},
            }
        }

    def save_manifest(self):
        with open(self.run_dir / "artifact_manifest.json", "w") as f:
            json.dump(self.manifest, f, indent=2)

    def load_manifest(self):
        manifest = json.load(open(self.run_dir / "artifact_manifest.json"))
        return manifest
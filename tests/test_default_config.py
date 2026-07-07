from pathlib import Path

import yaml


def test_default_training_artifacts_share_one_run_directory():
    config_path = Path("config/training.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    paths = config["paths"]
    artifact_dirs = [
        Path(paths["models_dir"]),
        Path(paths["reports_dir"]),
        Path(paths["outputs_dir"]),
    ]

    assert all(path.parts[:2] == ("runs", "latest") for path in artifact_dirs)
    assert {path.name for path in artifact_dirs} == {"models", "reports", "outputs"}

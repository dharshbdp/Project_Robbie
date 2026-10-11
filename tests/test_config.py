"""Tests for configuration loading and validation."""

from pathlib import Path
from robbie.config import RobbieConfig, CONFIG_DIR

def test_default_config():
    config = RobbieConfig()
    assert config.project_name == "Project Robbie"
    assert config.version == "0.2.0"
    assert config.seed == 42
    assert config.device == "auto"

def test_yaml_config_loading():
    yaml_file = CONFIG_DIR / "default.yaml"
    config = RobbieConfig.from_yaml(yaml_file)
    assert config.project_name == "Project Robbie"
    assert config.version == "0.2.0"
    assert config.seed == 42

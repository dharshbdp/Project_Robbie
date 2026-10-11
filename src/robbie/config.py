"""Configuration management for Project Robbie."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CONFIG_DIR = PROJECT_ROOT / "configs"

@dataclass
class RobbieConfig:
    """Master configuration for Robbie."""
    project_name: str = "Project Robbie"
    version: str = "0.2.0"
    seed: int = 42
    device: str = "auto"
    data_dir: Path = DATA_DIR
    checkpoint_dir: Path = CHECKPOINT_DIR
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "RobbieConfig":
        """Load configuration from a YAML file."""
        if not yaml_path.exists():
            return cls()
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(
            project_name=data.get("project_name", "Project Robbie"),
            version=data.get("version", "0.0.0"),
            seed=data.get("seed", 42),
            device=data.get("device", "auto"),
            extra=data.get("extra", {}),
        )

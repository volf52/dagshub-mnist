import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass
class Config:
    # The path of the dir containing config file. Could be Path(), which would depend on caller's dir
    base_dir: Path = Path(".").resolve().absolute()
    data_dir: Path = base_dir.joinpath("data")

    def populate(self, config_file: Path):
        assert config_file.exists()
        assert config_file.suffix == ".json"

        with config_file.open("r") as f:
            config = json.load(f)

        self.base_dir = Path(config.get("base_dir", self.base_dir))
        self.base_dir = self.base_dir.resolve().absolute()

        self.data_dir = (
            Path(config.get("data_dir", self.base_dir.joinpath("data")))
            .resolve()
            .absolute()
        )

        return self


@lru_cache(maxsize=1)
def get_config():
    return Config()

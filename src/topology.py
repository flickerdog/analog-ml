from importlib import import_module
from pathlib import Path

import yaml

_CONFIG = Path(__file__).with_name("topologies.yaml")


def load_topology(name):
    with _CONFIG.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    topologies = config.get("topologies", {})
    if name not in topologies:
        raise ValueError(f"unknown topology {name!r}; choose from: {', '.join(topologies)}")
    module = import_module(topologies[name]["module"])
    for attribute in ("RANGES", "DATA_FILE", "MODEL_FILE", "SCENARIO_FILE", "evaluate"):
        if not hasattr(module, attribute):
            raise ValueError(f"topology {name!r} is missing required attribute {attribute}")
    return module


def topology_names():
    with _CONFIG.open(encoding="utf-8") as file:
        return tuple(yaml.safe_load(file).get("topologies", {}))

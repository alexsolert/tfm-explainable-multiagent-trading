from pathlib import Path

from qqq_agents.config import load_config


def test_base_configuration_is_valid() -> None:
    config = load_config(Path("configs/base.yaml"))

    assert config.data.ticker == "QQQ"
    assert config.experiment.test_start == "2023-01-01"
    assert sum(config.coordinator.weights.values()) == 1.0
    assert config.llm.enabled is False

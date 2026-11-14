from __future__ import annotations

from experiments.driftlab.config import ExperimentConfig
from experiments.run import _apply_device_override


def _config(models: list[dict]) -> ExperimentConfig:
    return ExperimentConfig(
        exp_id="device-test",
        title="",
        description="",
        streams=[],
        models=models,
        detectors=[],
        detector_hosts=[],
        detector_channels=[],
        metrics={},
        seeds=[0],
        acceptance={},
    )


def test_device_override_only_touches_deep_models():
    config = _config(
        [
            {"id": "df", "type": "driftformer", "params": {"device": "cpu", "d_model": 64}},
            {"id": "dr", "type": "deepriver_rolling", "params": {"window_size": 500}},
            {"id": "ht", "type": "river", "params": {"class": "HoeffdingTreeClassifier"}},
        ]
    )
    _apply_device_override(config, "cuda")
    assert config.models[0]["params"]["device"] == "cuda"
    assert config.models[1]["params"]["device"] == "cuda"
    assert "device" not in config.models[2]["params"]


def test_device_override_creates_missing_params():
    config = _config([{"id": "dr", "type": "deepriver_rolling"}])
    _apply_device_override(config, "auto")
    assert config.models[0]["params"]["device"] == "auto"

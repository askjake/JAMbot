from pathlib import Path

from app.agent.complexity_detector import (
    choose_model_role_for_context,
    routing_metrics,
    should_use_complex_role_for_context,
)


def test_model_routing_metrics_are_provider_neutral(monkeypatch):
    monkeypatch.setattr("app.agent.complexity_detector.settings.AUTO_MODEL_ROUTING_ENABLED", True, raising=False)
    monkeypatch.setattr("app.agent.complexity_detector.settings.MODEL_ROUTING_COMPLEXITY_THRESHOLD", 3, raising=False)
    routing_metrics.reset()
    role = choose_model_role_for_context("follow protocol and investigate backend runtime traceback with pytest")
    stats = routing_metrics.get_stats()
    assert role == "complex"
    assert "complex_requests" in stats
    assert "primary_requests" in stats
    assert "complex_percentage" in stats
    assert "opus_requests" not in stats
    assert "sonnet_requests" not in stats
    assert "auto_opus_enabled" not in stats
    assert should_use_complex_role_for_context("debug and refactor a distributed backend with code") is True


def test_active_import_uses_model_routing_router():
    main = Path("app/main.py").read_text()
    router = Path("app/model_routing_metrics_router.py").read_text()
    assert "model_routing_metrics_router" in main
    assert "opus_metrics_router" not in main
    assert "/internal/model-routing-stats" in router

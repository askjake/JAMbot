from pathlib import Path
import importlib.util
import sys


def _module():
    path = Path(__file__).parents[1] / "scripts" / "security" / "scan_source_secrets.py"
    spec = importlib.util.spec_from_file_location("source_secret_scanner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_scanner_reports_paths_without_secret_values(tmp_path: Path):
    module = _module()
    secret = "gl" + "pat-testtoken1234567890"
    (tmp_path / "bad.txt").write_text(f"TOKEN={secret}\n")
    findings = module.scan(tmp_path)
    assert len(findings) == 1
    assert findings[0].path == "bad.txt"
    assert findings[0].line == 1
    assert findings[0].classification == "LIVE_CREDENTIAL_CANDIDATE"
    assert secret not in repr(findings[0])


def test_scanner_classifies_variableized_urls_and_test_fixtures(tmp_path: Path):
    module = _module()
    (tmp_path / "pipeline.yml").write_text(
        "git clone https://${USER}:${TOKEN}@gitlab.example/repo.git\n"
    )
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "fixture.py").write_text('value = "AK' + 'IAABCDEFGHIJKLMNOP"\n')
    findings = module.scan(tmp_path)
    classes = {item.classification for item in findings}
    assert "VARIABLEIZED_CREDENTIAL_REFERENCE" in classes
    assert "TEST_FIXTURE" in classes

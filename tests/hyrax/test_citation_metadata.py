from pathlib import Path
import tomllib


def test_citation_cff_exists_and_has_required_fields():
    """Ensure CITATION.cff exists at the repo root and contains required metadata keys."""
    cff_path = Path(__file__).resolve().parents[2] / "CITATION.cff"
    assert cff_path.exists()

    cff_text = cff_path.read_text(encoding="utf-8")
    lines = cff_text.splitlines()
    required_keys = [
        "cff-version:",
        "title:",
        "message:",
        "type:",
        "authors:",
        "repository-code:",
        "license:",
    ]
    for key in required_keys:
        assert any(line.lstrip().startswith(key) for line in lines), f"Required key '{key}' not found at start of any line in CITATION.cff"


def test_citation_compass_config_exists_and_has_curated_dependencies():
    """Ensure citation_compass.toml exists and lists key curated dependency entries."""
    config_path = Path(__file__).resolve().parents[2] / "citation_compass.toml"
    assert config_path.exists()

    config_text = config_path.read_text(encoding="utf-8")
    config_data = tomllib.loads(config_text)

    # Validate project metadata
    project = config_data.get("project")
    assert isinstance(project, dict)
    assert project.get("name") == "hyrax"

    # Validate curated dependency entries
    dependencies = config_data.get("dependencies")
    assert isinstance(dependencies, list)
    assert any(dep.get("package") == "torch" for dep in dependencies)
    assert any(dep.get("package") == "astropy" for dep in dependencies)

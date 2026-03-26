from pathlib import Path


def test_citation_cff_exists_and_has_required_fields():
    cff_path = Path(__file__).resolve().parents[2] / "CITATION.cff"
    assert cff_path.exists()

    cff_text = cff_path.read_text(encoding="utf-8")
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
        assert key in cff_text


def test_citation_compass_config_exists_and_has_curated_dependencies():
    config_path = Path(__file__).resolve().parents[2] / "citation_compass.toml"
    assert config_path.exists()

    config_text = config_path.read_text(encoding="utf-8")

    required_tokens = [
        "[project]",
        'name = "hyrax"',
        "[[dependencies]]",
        'package = "torch"',
        'package = "astropy"',
    ]

    for token in required_tokens:
        assert token in config_text


def test_citation_report_script_exists_and_docs_match_command():
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "print_citation_compass_report.py"
    assert script_path.exists()

    readme_text = (root / "README.md").read_text(encoding="utf-8")
    assert "python scripts/print_citation_compass_report.py --config citation_compass.toml" in readme_text

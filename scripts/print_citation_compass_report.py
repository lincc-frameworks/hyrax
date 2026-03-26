#!/usr/bin/env python3
"""Render Hyrax citation metadata using the citation-compass Python API.

This wrapper exists because citation-compass currently ships as a library,
not as a ``python -m citation_compass`` CLI entrypoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from citation_compass.citation import cite_inline, get_all_citations


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def parse_citation_compass_toml(config_path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Parse the small subset of TOML used by ``citation_compass.toml``.

    The repository config intentionally uses only simple string values in a
    single ``[project]`` table and repeated ``[[dependencies]]`` tables, so a
    lightweight parser is sufficient here and avoids requiring an additional
    TOML dependency for this helper script.
    """
    project: dict[str, str] = {}
    dependencies: list[dict[str, str]] = []
    current_dependency: dict[str, str] | None = None
    section: str | None = None

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "[project]":
            section = "project"
            current_dependency = None
            continue
        if line == "[[dependencies]]":
            section = "dependencies"
            current_dependency = {}
            dependencies.append(current_dependency)
            continue
        if "=" not in line:
            continue

        key, value = [part.strip() for part in line.split("=", 1)]
        value = _strip_quotes(value)

        if section == "project":
            project[key] = value
        elif section == "dependencies" and current_dependency is not None:
            current_dependency[key] = value

    return project, dependencies


def register_citations(config_path: Path) -> None:
    """Register the Hyrax and dependency citations described in the config file."""
    project, dependencies = parse_citation_compass_toml(config_path)

    project_name = project.get("name", "hyrax")
    repository = project.get("repository", "")
    primary_citation = project.get("primary_citation", "CITATION.cff")

    cite_inline(
        f"{project_name}.software",
        (
            f"Project: {project_name}\n"
            f"Cite the software using {primary_citation}.\n"
            f"Repository: {repository}"
        ),
    )

    for dependency in dependencies:
        package = dependency.get("package", "unknown-package")
        reason = dependency.get("reason", "No reason provided.")
        url = dependency.get("url", "")
        citation_text = f"Dependency: {package}\nReason in Hyrax: {reason}"
        if url:
            citation_text += f"\nProject URL: {url}"
        cite_inline(f"dependency.{package}", citation_text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Hyrax citation metadata using citation-compass.")
    parser.add_argument(
        "--config",
        default="citation_compass.toml",
        help="Path to the Citation Compass config file (default: citation_compass.toml).",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Citation Compass config file not found: {config_path}")

    register_citations(config_path)
    for citation in get_all_citations():
        print(citation)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

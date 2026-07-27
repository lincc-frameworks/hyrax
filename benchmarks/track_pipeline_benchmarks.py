import json
import os
from pathlib import Path

"""This module is used to track the performance of the full pipeline with asv."""


def track_test():
    """
    Load benchmark results from a JSON file under the ASV results directory and
    return the requested slowdown metric.
    """
    base_dir = Path(__file__).resolve().parent / "gondor_results"

    json_path = base_dir / "2026-07-24T22:08:39.409085+00:00.json"
    if not json_path.exists():
        return 11111111111111

    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return payload.get("train_slowdown")

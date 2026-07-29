import json
from pathlib import Path


class PipelineBenchmarkTracker:
    """Loads benchmark results once and return metrics for ASV."""

    def setup_cache(self):
        """
        Setup for loading self-hosted benchmark results. Opens the json file and save it as payload.
        """
        # TODO: "gondor_results" is the directory to store results from self-hosted runner on gondor
        base_dir = Path(__file__).resolve().parent / "gondor_results"
        # TODO: currently hard coded to tesy file loading only
        self.json_path = base_dir / "2026-07-24T22:08:39.409085+00:00.json"

        if not self.json_path.exists():
            print(f"Benchmark results not found: {self.json_path}")
            return {}

        with self.json_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


    def track_train_slowdown(self, payload):
        """
        Track and return the train slowdown factor between Hyrax and PyTorch.
        """
        return payload.get("train_slowdown")

    def track_infer_slowdown(self, payload):
        """
        Track and return the infer performance slowdown factor between Hyrax and PyTorch.
        """
        return payload.get("infer_slowdown")

    def track_total_slowdown(self, payload):
        """
        Track and return the overall performance slowdown factor between Hyrax and PyTorch.
        The metric combines training and inference time for both frameworks.
        """
        return payload.get("total_slowdown")


def track_test():
    """
    Trial function, kept for testing.
    Load benchmark results from a JSON file under the ASV results directory and
    return the requested slowdown metric.
    Should behave the same as `track_train_slowdown` above.
    """
    # Hard-coded path
    base_dir = Path(__file__).resolve().parent / "gondor_results"
    json_path = base_dir / "2026-07-24T22:08:39.409085+00:00.json"
    if not json_path.exists():
        return 11111111111111

    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return payload.get("train_slowdown")

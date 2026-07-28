import json
import os
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

        self.payload = None
        if not self.json_path.exists():
            print(f"Benchmark results not found: {self.json_path}")
            self.payload = {}
            return

        with self.json_path.open("r", encoding="utf-8") as handle:
            self.payload = json.load(handle)

        print(f"Loaded benchmark results from {self.json_path}")
        print(self.payload)

    def track_train_slowdown(self):
        return self.payload.get("train_slowdown")

    def track_infer_slowdown(self):
        return self.payload.get("infer_slowdown")

    def track_total_slowdown(self):
        return self.payload.get("total_slowdown")

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

# if __name__ == "__main__":
#     result = track_test()
#     print(f"Returned from track_test: {result}")

#     tracker = PipelineBenchmarkTracker()
#     print(tracker.track_train_slowdown())
#     print(tracker.track_infer_slowdown())
#     print(tracker.track_total_slowdown())

import os
import datetime
import json
import platform
from pathlib import Path

import torch
import hyrax

from timebench_utils import benchmark_repeated


CONFIG = {
    "train_fraction": 1.0,
    "epochs": 10,
    "batch_size": 512,
    "num_workers": 0,
    "lr": 0.01,
    "repeats": 5,
}


def get_device():
    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)

    return "cpu"


def main():
    pytorch_res = benchmark_repeated(
        "pytorch",
        **CONFIG,
    )

    hyrax_res = benchmark_repeated(
        "hyrax",
        **CONFIG,
    )

    result = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "hyrax_version": hyrax.__version__,
        "device": get_device(),

        **CONFIG,

        "hyrax_train_time": hyrax_res["train_time_mean"],
        "hyrax_train_std": hyrax_res["train_time_std"],

        "hyrax_infer_time": hyrax_res["infer_time_mean"],
        "hyrax_infer_std": hyrax_res["infer_time_std"],

        "hyrax_accuracy": hyrax_res["accuracy_mean"],

        "pytorch_train_time": pytorch_res["train_time_mean"],
        "pytorch_train_std": pytorch_res["train_time_std"],

        "pytorch_infer_time": pytorch_res["infer_time_mean"],
        "pytorch_infer_std": pytorch_res["infer_time_std"],

        "pytorch_accuracy": pytorch_res["accuracy_mean"],
    }

    result["train_slowdown"] = (result["hyrax_train_time"] / result["pytorch_train_time"])
    result["infer_slowdown"] = (result["hyrax_infer_time"] / result["pytorch_infer_time"])
    result["total_slowdown"] = ((result["hyrax_train_time"] + result["hyrax_infer_time"]) 
                              / (result["pytorch_train_time"] + result["pytorch_infer_time"]))

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    # save the result to json file
    output_dir = Path(os.environ.get("RESULT_PATH", "benchmarks/results"))
    # make sure the directory exist
    output_dir.mkdir(parents=True, exist_ok=True)

    result_file = output_dir / f"{result["timestamp"]}.json"

    with open(result_file, "w") as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print(f"RESULT_FILE={result_file.resolve()}")

if __name__ == "__main__":
    main()
import argparse
import datetime
import json
import os
import platform
import subprocess
from pathlib import Path

import torch
from timebench_hsc_utils import benchmark_repeated_hsc
from timebench_utils import benchmark_repeated

import hyrax

CIFAR_CONFIG = {
    "train_fraction": 1.0,
    "epochs": 5,
    "batch_size": 512,
    "num_workers": 0,
    "lr": 0.01,
    "repeats": 3,
}

HSC_CONFIG = {
    "train_fraction": 1.0,
    "epochs": 10,
    "batch_size": 128,
    "num_workers": 0,
    "lr": 0.01,
    "repeats": 3,
}

# name -> (repeated benchmark function, config, name of the quality metric in the summary dict)
BENCHMARKS = {
    "cifar": (benchmark_repeated, CIFAR_CONFIG, "accuracy"),
    "hsc": (benchmark_repeated_hsc, HSC_CONFIG, "recon_mse"),
}


def get_git_commit():
    """Get the current git commit hash of the repository."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def get_device():
    """Get the current device being used for computation."""
    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)

    return "cpu"


def run_benchmark(name, repeated_fn, config, metric_key):
    """
    Run one benchmark for both PyTorch and Hyrax and save the results to a JSON file.

    The file is written to ``<RESULT_PATH>/<short_commit>_<name>_<YYYYMMDD_HHMMSS>.json``, where
    ``RESULT_PATH`` defaults to ``benchmarks_self_hosted/results``. The absolute path is printed on
    a ``RESULT_FILE=`` line so a calling workflow can pick it up.

    Parameters
    ----------
    name : str
        Benchmark name, e.g. ``"cifar"`` or ``"hsc"``.
    repeated_fn : callable
        ``benchmark_repeated``-style function taking ``framework`` and ``**config``.
    config : dict
        Keyword arguments passed to ``repeated_fn``.
    metric_key : str
        Key of the quality metric in the summary returned by ``repeated_fn`` (``"accuracy"``,
        ``"recon_mse"``); it is recorded as ``hyrax_<metric_key>`` / ``pytorch_<metric_key>``.

    Returns
    -------
    Path
        The result file that was written.
    """
    pytorch_res = repeated_fn("pytorch", **config)
    hyrax_res = repeated_fn("hyrax", **config)

    result = {
        "benchmark": name,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "commit": get_git_commit(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "hyrax_version": hyrax.__version__,
        "device": get_device(),
        **config,
        "hyrax_train_time": hyrax_res["train_time_mean"],
        "hyrax_train_std": hyrax_res["train_time_std"],
        "hyrax_infer_time": hyrax_res["infer_time_mean"],
        "hyrax_infer_std": hyrax_res["infer_time_std"],
        f"hyrax_{metric_key}": hyrax_res[f"{metric_key}_mean"],
        "pytorch_train_time": pytorch_res["train_time_mean"],
        "pytorch_train_std": pytorch_res["train_time_std"],
        "pytorch_infer_time": pytorch_res["infer_time_mean"],
        "pytorch_infer_std": pytorch_res["infer_time_std"],
        f"pytorch_{metric_key}": pytorch_res[f"{metric_key}_mean"],
    }

    # Extra context that only the HSC benchmark reports.
    for key in ("use_cache", "num_samples"):
        for framework, res in (("hyrax", hyrax_res), ("pytorch", pytorch_res)):
            if key in res:
                result[f"{framework}_{key}"] = res[key]

    result["train_slowdown"] = result["hyrax_train_time"] / result["pytorch_train_time"]
    result["infer_slowdown"] = result["hyrax_infer_time"] / result["pytorch_infer_time"]
    result["total_slowdown"] = (result["hyrax_train_time"] + result["hyrax_infer_time"]) / (
        result["pytorch_train_time"] + result["pytorch_infer_time"]
    )

    short_result = {
        "benchmark": name,
        "device": get_device(),
        **config,
        "hyrax_train_time": hyrax_res["train_time_mean"],
        "pytorch_train_time": pytorch_res["train_time_mean"],
        "train_slowdown": result["train_slowdown"],
        "hyrax_infer_time": hyrax_res["infer_time_mean"],
        "pytorch_infer_time": pytorch_res["infer_time_mean"],
        "infer_slowdown": result["infer_slowdown"],
    }

    print(
        json.dumps(
            short_result,
            indent=2,
        )
    )

    # save the result to json file
    output_dir = Path(os.environ.get("RESULT_PATH", "benchmarks_self_hosted/results"))
    # make sure the directory exist
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.fromisoformat(result["timestamp"])
    # filename format: <short_commit>_<name>_<YYYYMMDD_HHMMSS>.json
    filename = f"{result['commit'][:8]}_{name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    result_file = output_dir / filename

    with open(result_file, "w") as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print(f"RESULT_FILE={result_file.resolve()}")
    return result_file


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hyrax vs. plain PyTorch runtime benchmarks.")
    parser.add_argument(
        "--benchmark",
        choices=[*BENCHMARKS, "all"],
        default="cifar",
        help="Which benchmark to run: 'cifar' (CNN on CIFAR-10, default), "
        "'hsc' (HyraxAutoencoderV2 on HSC1k cutouts), or 'all'.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """
    Run the selected benchmark(s) for both PyTorch and Hyrax and save each result to a JSON file.
    One ``RESULT_FILE=`` line is printed per benchmark.
    """
    args = parse_args(argv)
    names = list(BENCHMARKS) if args.benchmark == "all" else [args.benchmark]

    for name in names:
        repeated_fn, config, metric_key = BENCHMARKS[name]
        run_benchmark(name, repeated_fn, config, metric_key)


if __name__ == "__main__":
    main()

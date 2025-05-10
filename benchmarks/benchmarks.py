"""Two sample benchmarks to compute runtime and memory usage.

For more information on writing benchmarks:
https://asv.readthedocs.io/en/stable/writing_benchmarks.html."""

import subprocess

from hyrax import example_benchmarks


def time_computation():
    """Time computations are prefixed with 'time'."""
    example_benchmarks.runtime_computation()


def mem_list():
    """Memory computations are prefixed with 'mem' or 'peakmem'."""
    return example_benchmarks.memory_computation()


def time_import():
    """
    time how long it takes to import our package. This should stay relatively fast.

    Note, the actual import time will be slightly lower than this on a comparable system
    However, high import times do affect this metric proportionally.
    """
    result = subprocess.run(["python", "-c", "import hyrax"])
    assert result.returncode == 0

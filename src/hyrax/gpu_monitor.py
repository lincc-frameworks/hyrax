import os
import time
from dataclasses import dataclass
from subprocess import PIPE, Popen
from threading import Thread
from hyrax.tensorboardx_logger import getTensorboardLogger


class GpuMonitor(Thread):
    """General GPU monitor that runs in a separate thread and logs GPU metrics
    to Tensorboard.
    """

    def __init__(self, interval_seconds=1):
        super().__init__()
        self.stopped = False
        self.delay = interval_seconds  # Seconds between calls to GPUtil
        self.start_time = time.time()
        self.tensorboard_logger = getTensorboardLogger()
        self.start()

    def run(self):
        """Run loop that logs GPU metrics every `self.delay` seconds."""
        while not self.stopped:
            gpus = get_gpu_info()
            step = time.time() - self.start_time
            for gpu in gpus:
                gpu_name = f"GPU_{gpu.id}"
                self.tensorboard_logger.add_scalar(f"{gpu_name}/load", gpu.load * 100, step)
                self.tensorboard_logger.add_scalar(
                    f"{gpu_name}/memory_utilization", gpu.memory_util * 100, step
                )
            time.sleep(self.delay)

    def stop(self):
        """Stop the monitoring thread."""
        self.stopped = True


"""The following is based on GPUtil. It has been striped down to only the
functionality needed for Hyrax. The original code can be found here:
https://github.com/anderskm/gputil
"""


@dataclass
class GPU:
    """Holds the GPU metrics retrieved from nvidia-smi."""

    id: int
    load: float
    memory_total: float
    memory_used: float

    @property
    def memory_util(self):
        """Return the memory utilization of the GPU."""
        return self.memory_used / self.memory_total


def safe_float_cast(str_number):
    """Convert a string into a float handling the case of `nan`.

    Parameters
    ----------
    str_number : str
        The string to convert to a float.

    Returns
    -------
    float
        The converted float.
    """
    try:
        number = float(str_number)
    except ValueError:
        number = float("nan")
    return number


def get_gpu_info():
    """Get the GPU utilization and memory usage for all GPUs on the system using
    nvidia-smi. Returns a list of GPU objects."""

    try:
        p = Popen(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            stdout=PIPE,
        )
        stdout, stderror = p.communicate()
    except:  # noqa: E722
        return []
    output = stdout.decode("UTF-8")

    # Parse output
    lines = output.split(os.linesep)

    num_devices = len(lines) - 1
    gpus = []
    for g in range(num_devices):
        line = lines[g]
        vals = line.split(", ")
        for i in range(4):
            if i == 0:
                device_ids = int(vals[i])
            elif i == 1:
                gpu_util = safe_float_cast(vals[i]) / 100
            elif i == 2:
                mem_total = safe_float_cast(vals[i])
            elif i == 3:
                mem_used = safe_float_cast(vals[i])

        gpus.append(GPU(device_ids, gpu_util, mem_total, mem_used))
    return gpus

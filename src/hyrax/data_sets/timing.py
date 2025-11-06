import contextlib
import functools
import time
from typing import Callable


def report_duration_to_tensorboard(report_every: int = 100):
    """
    Decorator for dataset instance methods that reports total run time to
    tensorboard via `self.log_duration_tensorboard(method_name, start_ns)`.

    Usage:
      @report_duration_to_tensorboard
      def get_image(...): ...

      @report_duration_to_tensorboard(50)
      def get_image(...): ...

    The emitted metric name will always be the wrapped method's __name__.
    """
    # Support using the decorator without parentheses: @report_duration_to_tensorboard
    if callable(report_every):
        fn = report_every  # type: ignore
        report_every = 100
        return report_duration_to_tensorboard(report_every)(fn)

    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            start_ns = time.monotonic_ns()
            result = fn(self, *args, **kwargs)

            # ensure counter storage on the instance
            counters = getattr(self, "_timing_counters", None)
            if counters is None:
                counters = {}
                self._timing_counters = counters

            evt_name = fn.__name__
            cnt = counters.get(evt_name, 0) + 1
            counters[evt_name] = cnt

            if report_every > 0 and (cnt % report_every == 0):
                log_fn = getattr(self, "log_duration_tensorboard", None)
                if callable(log_fn):
                    with contextlib.suppress(Exception):
                        # pass the method name to the logger
                        log_fn(evt_name, start_ns)
            return result

        return wrapper

    return decorator

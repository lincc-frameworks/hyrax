# Usage
#
# from hyrax.tensorboardx_logger import initTensorboardLogger, getTensorboardLogger
#
# initTensorboardLogger(log_dir="runs/exp1")  # configure as needed
# tb_logger = getTensorboardLogger()
#
# ...in code...
#
# tb_logger.log_scalar(...)
# tb_logger.log_duration(...)

import time
import inspect
from tensorboardX import SummaryWriter


tensorboardx_logger = None
tensorboard_start_ns = 0

def getTensorboardLogger():
    return HyraxSummaryWriter()

def initTensorboardLogger(**kwargs):
    global tensorboardx_logger
    global tensorboard_start_ns

    closeTensorboardLogger()
    tensorboardx_logger = SummaryWriter(**kwargs)
    tensorboard_start_ns = time.monotonic_ns()

def closeTensorboardLogger():
    global tensorboardx_logger
    global tensorboard_start_ns
    
    if tensorboardx_logger is not None:
        tensorboardx_logger.close()
        tensorboardx_logger = None
        tensorboard_start_ns = 0

class HyraxSummaryWriter:
    """
    This is a wrapper class around TensorboardX SummaryWriter that allows definition
    of convenience methods for commonly-used logging.

    __dir__ and __getattr__ pass through function calls to the underlying tensorboardX SummaryWriter if
    it exists. Otherwise empty/noop objects are returned. We don't use inheritance here because we want
    consumers to not have to think about initialization order concerns, yet have a handle to a pile of 
    functions that all log to the one true tensorboard instance (if it exists)

    All functions defined here need to be noops when global tensorboardx_logger is None.

    We have the capacity to place information on instances of this class (e.g. a name prefix)
    but its not implemented. One major issue is providing continuity of interface with tensorboard's 
    functions that don't recognize a name prefix. For now, the fully qualified tensorboard name of the
    data is the common interface, since that follows what tensorboard functions expect.

    """
    def log_duration_ts(self, name: str, start_time: int):
        """
        Log a duration to tensorboardX as a time series if configured.
        
        Caller provides the start of the duration in time.monotonic_ns
        End of the duration is assumed to be the moment the function is called.

        Parameters
        ----------
        name : str
            The name of the scalar to log
        start_time : int
            Start time in nanoseconds from time.monotonic_ns() 
        """
        now = time.monotonic_ns()
        if tensorboardx_logger:
            since_tensorboard_start_us = (start_time - tensorboard_start_ns) / 1.0e3
            duration_s = (now - start_time) / 1.0e9
            self.log_scalar_ts(name, duration_s, since_tensorboard_start_us)

    def log_scalar_ts(self, name: str, scalar, since_tensorboard_start_ns=None):
        """
        Log a scalar to tensorboardX as a time series if configured.

        Parameters
        ----------
        name : str
            The name of the scalar to log
        scalar: Any
            The value to log. Really ought to be a number.
        since_tensorboard_start_ns : int, Optional
            Log time in nanoseconds from the beginning of tensorboard logging.
            If not provided, this will be calculated at the moment this function is called
        """
        now = time.monotonic_ns()
        if tensorboardx_logger:
            since_tensorboard_start_ns = (now - tensorboard_start_ns) if since_tensorboard_start_ns is None \
                else since_tensorboard_start_ns

            since_tensorboard_start_us = since_tensorboard_start_ns / 1.0e3
            tensorboardx_logger.add_scalar(name, scalar, since_tensorboard_start_us)

    def __dir__(self):
        methods = [ i for i in dir(HyraxSummaryWriter) \
                    if inspect.isfunction(getattr(HyraxSummaryWriter, i)) ]
        
        return sorted(set(methods + dir(SummaryWriter)))

    def __getattr__(self, name):
        # Reminder of python behavior:
        # __getattr__ is called when there's an AttributeError looking up
        # an attribute on instances of HyraxSummaryWriter.
        # 
        # It's job is to either return an object or raise AttributeError

        # If we have a tensorboardX logger, just pass through access there
        if tensorboardx_logger is not None:
            return getattr(tensorboardx_logger, name)
        
        # Otherwise if its a valid access of SummaryWriter's methods or members
        elif name in dir(SummaryWriter):

            # Function access returns a noop function
            if inspect.isfunction(getattr(SummaryWriter, name)):
                def noop(*args, **kwargs):
                    pass
                return noop
            # member access returns None
            else:
                return None
        # All other access is an AttributeError
        else:
            raise AttributeError(name)
    
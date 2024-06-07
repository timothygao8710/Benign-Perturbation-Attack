import os
import sys
from contextlib import contextmanager

@contextmanager
def suppress_output(Verbose):
    if not Verbose:
        # Redirect standard output and standard error to devnull
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = devnull
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
    else:
        yield
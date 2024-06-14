import os
import sys
from contextlib import contextmanager
import ast
import random

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
        
        
def select_random_subset(f, subset_size=1000):
    """
    Select a random subset of specified size from the list f.

    Parameters:
    f (list): The list to select from.
    subset_size (int): The size of the subset to select.

    Returns:
    list: A random subset of the original list.
    """
    if subset_size > len(f):
        raise ValueError("Subset size is larger than the size of the input list")
        
    return random.sample(f, subset_size)


def strToDict(Str):
    return ast.literal_eval(Str)
import os
import io
import gzip
import json
import warnings
from functools import partial, wraps
from typing import Any, Callable
import torch


def rank_0_only(fn: Callable) -> Callable:
    """Call a function only on rank 0 in distributed settings.

    Meant to be used as an decorator.

    """

    @wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Any:
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = int(os.getenv("RANK", 0))

        if rank == 0:
            return fn(*args, **kwargs)

        return None

    return wrapped_fn


print_rank_0 = rank_0_only(partial(print, flush=True))
warn_rank_0 = rank_0_only(warnings.warn)


def jload(f, mode="r", object_pairs_hook=None):
    """Load a .json file into a dictionary."""
    if not isinstance(f, io.IOBase):
        if f.endswith(".json.gz"):
            f = gzip.open(f, mode="rt")
        else:
            f = open(f, mode=mode)
    jdict = json.load(f, object_pairs_hook=object_pairs_hook)
    f.close()
    return jdict

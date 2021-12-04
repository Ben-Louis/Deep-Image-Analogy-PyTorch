from .PatchMatchCpu import init_nnf, upSample_nnf
from .PatchMatchCpu import propagate as propagate_cpu
from .PatchMatchCpu import avg_vote as avg_vote_cpu

propagate_func = {"cpu": propagate_cpu}
avg_vote_func = {"cpu": avg_vote_cpu}

try:
    from .PatchMatchCuda import propagate_gpu, avg_vote_gpu
    propagate_func["gpu"] = propagate_gpu
    avg_vote_func["gpu"] = avg_vote_gpu
except ImportError:
    pass

__all__ = ["init_nnf", "upSample_nnf", "propagate_func", "avg_vote_func"]
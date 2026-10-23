"""Exit codes of ``nstrain train`` (the supervisor acts on them)."""
OK = 0                  # done, stopped, the line's data exhausted
FAILURE = 1             # an error: the supervisor relaunches (counted when the line made no progress)
GPU_BUSY = 3            # another process uses the GPU: wait until it is free, not a failure
LINE_BUSY = 4           # another trainer holds this line (e.g. one still writing its last checkpoint): wait for it
NEEDS_OPERATOR = 5      # stopped with a checkpoint for a condition a restart cannot fix (disk space): do not relaunch
GPU_FREE = 6            # trained beside other processes and they left: relaunch with the GPU to itself

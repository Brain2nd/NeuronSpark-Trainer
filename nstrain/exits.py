"""Exit codes of ``nstrain train`` (the supervisor acts on them)."""
OK = 0                  # done, stopped, the line's data exhausted
FAILURE = 1             # an error: the supervisor relaunches (counted when the line made no progress)
GPU_BUSY = 3            # another process uses the GPU, or holds its lease: wait until it is free, not a failure
LINE_BUSY = 4           # another trainer holds this line (e.g. one still writing its last checkpoint): wait for it
NEEDS_OPERATOR = 5      # a condition a restart cannot fix (disk space, a failed start check, the data off their plan,
                        # another trainer reading the data line): do not relaunch

SIGNAL_STOP = "stopped by a signal"     # the last heartbeat stage of a trainer stopped by a signal (exit 0)

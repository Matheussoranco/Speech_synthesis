"
Secure subprocess management for contained backend
"

import os
import sys
import fcntl

def secure_backend_drain_fd() -> None:
    """
    Re-arm PEP 446 (CLOEXEC) on the backend drain file descriptor.
    This prevents third-party close_fds=False children from holding the barrier.
    """
    drain_fd = int(os.environ.get("VOICESTUDIO_BACKEND_DRAIN_FD", "-1"))
    if drain_fd >= 0:
        try:
            flags = fcntl.fcntl(drain_fd, fcntl.F_GETFD)
            fcntl.fcntl(drain_fd, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)
        except Exception:
            pass

def supervisor_main(argv: list[str]) -> int:
    """
    Entry point for the supervisor subprocess.
    Used when the frozen backend is launched with --supervise.
    """
    # This would manage child processes in the frozen build
    print("[supervisor] Not implemented in development mode")
    return 0

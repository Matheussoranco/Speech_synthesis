"""
Secure subprocess management for contained backend
"""

import os
import sys

def secure_backend_drain_fd() -> None:
    """
    Re-arm PEP 446 (CLOEXEC) on the backend drain file descriptor.
    This prevents third-party close_fds=False children from holding the barrier.
    """
    drain_fd = int(os.environ.get("VOICESTUDIO_BACKEND_DRAIN_FD", "-1"))
    if drain_fd >= 0:
        try:
            os.set_inheritable(drain_fd, False)
        except OSError as error:
            raise RuntimeError("Could not protect the backend drain descriptor") from error

def supervisor_main(argv: list[str]) -> int:
    """
    Entry point for the supervisor subprocess.
    Used when the frozen backend is launched with --supervise.
    """
    # This would manage child processes in the frozen build
    print("[supervisor] Not implemented in development mode")
    return 0

"""
Windows-specific subprocess handling to suppress console windows
"""

import sys
import subprocess
import inspect

_original_popen = None

def install() -> None:
    """Patch subprocess.Popen to suppress console windows on Windows"""
    global _original_popen
    
    if sys.platform != "win32":
        return
    
    if _original_popen is not None:
        return  # Already installed
    
    _original_popen = subprocess.Popen
    
    flag_position = list(inspect.signature(_original_popen).parameters).index("creationflags")

    class QuietPopen(_original_popen):
        # asyncio subclasses Popen during import. Preserve its class contract.
        def __init__(self, *args, **kwargs):
            if len(args) > flag_position:
                args = list(args)
                args[flag_position] |= subprocess.CREATE_NO_WINDOW
            else:
                kwargs["creationflags"] = kwargs.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW
            super().__init__(*args, **kwargs)

    # subprocess.run/call/check_* already resolve this class at call time.
    subprocess.Popen = QuietPopen

"
Windows-specific subprocess handling to suppress console windows
"

import sys
import subprocess

_original_popen = None

def install() -> None:
    """Patch subprocess.Popen to suppress console windows on Windows"""
    global _original_popen
    
    if sys.platform != "win32":
        return
    
    if _original_popen is not None:
        return  # Already installed
    
    _original_popen = subprocess.Popen
    
    def patched_popen(*args, **kwargs):
        # Add CREATE_NO_WINDOW flag
        creationflags = kwargs.get("creationflags", 0)
        kwargs["creationflags"] = creationflags | subprocess.CREATE_NO_WINDOW
        return _original_popen(*args, **kwargs)
    
    subprocess.Popen = patched_popen
    # Also patch subprocess.run, call, check_call, etc.
    for name in ["run", "call", "check_call", "check_output"]:
        if hasattr(subprocess, name):
            original = getattr(subprocess, name)
            def make_wrapper(orig):
                def wrapper(*args, **kwargs):
                    creationflags = kwargs.get("creationflags", 0)
                    kwargs["creationflags"] = creationflags | subprocess.CREATE_NO_WINDOW
                    return orig(*args, **kwargs)
                return wrapper
            setattr(subprocess, name, make_wrapper(original))

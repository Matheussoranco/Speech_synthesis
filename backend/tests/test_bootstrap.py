import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_backend_module_parses():
    for source in ROOT.rglob('*.py'):
        ast.parse(source.read_text(encoding='utf-8-sig'), filename=str(source))


def test_drain_descriptor_is_not_inherited(tmp_path, monkeypatch):
    module = load_module('contained_test', 'core/contained_subprocess.py')
    with (tmp_path / 'drain').open('wb') as file:
        os.set_inheritable(file.fileno(), True)
        monkeypatch.setenv('VOICESTUDIO_BACKEND_DRAIN_FD', str(file.fileno()))
        module.secure_backend_drain_fd()
        assert not os.get_inheritable(file.fileno())


def test_windows_subprocess_class_contract(monkeypatch):
    module = load_module('win_subprocess_test', 'core/win_subprocess.py')
    calls = []

    class Original:
        def __init__(self, args, creationflags=0):
            calls.append(creationflags)

    monkeypatch.setattr(subprocess, 'Popen', Original)
    monkeypatch.setattr(subprocess, 'CREATE_NO_WINDOW', 8, raising=False)
    monkeypatch.setattr(sys, 'platform', 'win32')
    module.install()
    assert issubclass(subprocess.Popen, Original)

    class AsyncioCompatible(subprocess.Popen):
        pass

    AsyncioCompatible(['test'], creationflags=2)
    AsyncioCompatible(['test'], 4)
    assert calls == [10, 12]
    installed = subprocess.Popen
    module.install()
    assert subprocess.Popen is installed

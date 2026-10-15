"""Real Windows speech synthesis using installed voices; no model downloads."""
import json
import os
from pathlib import Path
import subprocess
import sys


class SpeechUnavailable(RuntimeError):
    pass


_SCRIPT = r'''
$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = [Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [Text.UTF8Encoding]::new()
$request = [Console]::In.ReadToEnd() | ConvertFrom-Json
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
  $voices = @($synth.GetInstalledVoices() | Where-Object Enabled | ForEach-Object {
    @{ name = $_.VoiceInfo.Name; language = $_.VoiceInfo.Culture.Name }
  })
  if ($request.action -eq 'voices') {
    ConvertTo-Json -InputObject $voices -Compress
  } else {
    $chosen = $null
    if ($request.voice -ne 'default') {
      $chosen = $voices | Where-Object { $_.name -eq $request.voice } | Select-Object -First 1
    } elseif ($request.language -eq 'auto') {
      $chosen = $voices | Select-Object -First 1
    } else {
      $chosen = $voices | Where-Object {
        $_.language -eq $request.language -or $_.language.StartsWith($request.language + '-')
      } | Select-Object -First 1
    }
    if (-not $chosen) { throw 'No installed voice matches the requested voice/language' }
    $synth.SelectVoice($chosen.name)
    $synth.Rate = [int]$request.rate
    $synth.SetOutputToWaveFile([string]$request.output)
    $synth.Speak([string]$request.text)
    $synth.SetOutputToNull()
    ConvertTo-Json -InputObject $chosen -Compress
  }
} finally { $synth.Dispose() }
'''


def _run(request: dict, timeout: int = 30):
    if sys.platform != 'win32':
        raise SpeechUnavailable('Windows System Speech is unavailable on this platform')
    executable = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
    try:
        response = subprocess.run(
            [str(executable), '-NoLogo', '-NoProfile', '-NonInteractive', '-Command', _SCRIPT],
            input=json.dumps(request), capture_output=True, encoding='utf-8',
            timeout=timeout, check=True, creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return json.loads(response.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise SpeechUnavailable('Installed speech voice could not be used') from error


def installed_voices() -> list[dict[str, str]]:
    return _run({'action': 'voices'})


def synthesize(text: str, output: Path, *, voice: str, language: str, speed: float, timeout: int):
    # System Speech exposes a native rate from -10 to 10, rather than an exact
    # time-stretch factor. The requested speed maps to that documented control.
    rate = max(-10, min(10, round((speed - 1.0) * 10)))
    return _run({'action': 'synthesize', 'text': text, 'output': str(output.resolve()),
                 'voice': voice, 'language': language, 'rate': rate}, timeout=timeout)

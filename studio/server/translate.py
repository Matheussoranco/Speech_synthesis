"""Optional text translation through a local Ollama server."""
import json
import re
import urllib.error
import urllib.request

from . import config


class TranslationUnavailable(RuntimeError):
    pass


def _request(path: str, payload: dict | None = None, timeout: float = 5.0) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(config.OLLAMA_URL + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        raise TranslationUnavailable(f"Ollama indisponível em {config.OLLAMA_URL}") from e


def models() -> list[str]:
    try:
        return [m["name"] for m in _request("/api/tags").get("models", [])]
    except TranslationUnavailable:
        return []


def _model() -> str:
    if config.OLLAMA_MODEL:
        return config.OLLAMA_MODEL
    names = models()
    if not names:
        raise TranslationUnavailable("Nenhum modelo do Ollama instalado para tradução")
    return names[0]


def translate_lines(lines: list[str], target: str, source: str = "") -> list[str]:
    """Translate a list of subtitle lines, keeping the same count and order."""
    model = _model()
    out: list[str] = []
    batch = 20
    for start in range(0, len(lines), batch):
        chunk = lines[start:start + batch]
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(chunk))
        prompt = (
            f"Translate each numbered line{' from ' + source if source else ''} into {target}. "
            "These are spoken subtitle lines; keep them natural and about the same length. "
            f"Reply with exactly {len(chunk)} lines in the form 'N. translation' and nothing else.\n\n{numbered}"
        )
        last = start + batch >= len(lines)
        resp = _request("/api/generate", {"model": model, "prompt": prompt, "stream": False, "think": False,
                                          "options": {"temperature": 0.2, "num_ctx": 4096},
                                          # free its VRAM for the voice model once done
                                          "keep_alive": 0 if last else "2m"}, timeout=600)
        text = re.sub(r"<think>.*?</think>", "", resp.get("response", ""), flags=re.S)
        found: dict[int, str] = {}
        for line in text.splitlines():
            m = re.match(r"\s*(\d+)[.)]\s*(.+)", line)
            if m:
                found.setdefault(int(m.group(1)), m.group(2).strip())
        out.extend(found.get(i + 1, original) for i, original in enumerate(chunk))
    return out

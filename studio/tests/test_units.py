"""Fast tests that need no model downloads."""
import os
import tempfile

os.environ.setdefault("STUDIO_DATA_DIR", tempfile.mkdtemp())

import numpy as np  # noqa: E402

from server import asr, audio, dubbing, engine, languages, store  # noqa: E402


def test_split_text_keeps_all_words_and_respects_limit():
    text = ("Primeira frase. Segunda frase! " * 30) + "\n\nNovo parágrafo aqui."
    chunks = engine.split_text(text, max_chars=120)
    assert all(len(c) <= 120 for c, _ in chunks)
    assert " ".join(c for c, _ in chunks).split() == text.split()
    assert chunks[-1][1] == engine.PAUSE_PARAGRAPH


def test_split_text_breaks_long_sentence_without_punctuation():
    text = "palavra " * 200
    chunks = engine.split_text(text, max_chars=100)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c, _ in chunks)
    assert " ".join(c for c, _ in chunks).split() == text.split()


def test_srt_format():
    srt = asr.to_srt([{"start": 0.0, "end": 1.5, "text": "Oi"}, {"start": 61.25, "end": 3725.0, "text": "Tchau"}])
    assert "1\n00:00:00,000 --> 00:00:01,500\nOi" in srt
    assert "2\n00:01:01,250 --> 01:02:05,000\nTchau" in srt


def test_language_lookup():
    assert languages.code_for("Portuguese") == "pt"
    assert languages.name_for("pt") == "Portuguese"
    assert languages.name_for("xx") == "xx"


def test_pick_reference_prefers_long_consecutive_speech():
    segs = [{"start": 0, "end": 1, "text": "a"}, {"start": 5, "end": 9, "text": "b"},
            {"start": 9, "end": 14, "text": "c"}, {"start": 30, "end": 31, "text": "d"}]
    start, end, text = dubbing._pick_reference(segs, 40)
    assert (start, end, text) == (5, 14, "b c")


def test_trim_silence():
    sr = 1000
    x = np.concatenate([np.zeros(500), np.ones(100) * 0.5, np.zeros(500)]).astype(np.float32)
    y = audio.trim_silence(x, sr, pad=0.01)
    assert 100 <= y.size <= 130


def test_store_roundtrip():
    v = store.add_voice("Teste", "a.wav", "olá")
    assert store.find_voice("teste")["id"] == v["id"]
    store.update_voice(v["id"], name="Outro")
    assert store.get_voice(v["id"])["name"] == "Outro"
    h = store.add_history("auto", "título", {"audio": "/files/outputs/x/a.mp3"}, {"text": "t"})
    assert store.get_history(h["id"])["files"]["audio"].endswith("a.mp3")
    store.delete_voice(v["id"])
    assert store.get_voice(v["id"]) is None

"""Languages offered in the UI. The voice model supports far more; any ISO code or
English name can also be sent to the API directly."""

LANGUAGES = [
    ("pt", "Portuguese", "Português"),
    ("en", "English", "English"),
    ("es", "Spanish", "Español"),
    ("fr", "French", "Français"),
    ("de", "German", "Deutsch"),
    ("it", "Italian", "Italiano"),
    ("nl", "Dutch", "Nederlands"),
    ("pl", "Polish", "Polski"),
    ("ru", "Russian", "Русский"),
    ("uk", "Ukrainian", "Українська"),
    ("tr", "Turkish", "Türkçe"),
    ("ar", "Arabic", "العربية"),
    ("hi", "Hindi", "हिन्दी"),
    ("ja", "Japanese", "日本語"),
    ("ko", "Korean", "한국어"),
    ("zh", "Chinese", "中文"),
    ("id", "Indonesian", "Bahasa Indonesia"),
    ("vi", "Vietnamese", "Tiếng Việt"),
    ("th", "Thai", "ไทย"),
    ("sv", "Swedish", "Svenska"),
    ("el", "Greek", "Ελληνικά"),
    ("he", "Hebrew", "עברית"),
]

_BY_NAME = {name.lower(): code for code, name, _ in LANGUAGES}
_BY_CODE = {code: name for code, name, _ in LANGUAGES}


def code_for(value: str) -> str:
    value = (value or "").strip()
    return _BY_NAME.get(value.lower(), value.lower())


def name_for(value: str) -> str:
    value = (value or "").strip()
    return _BY_CODE.get(value.lower(), value)


def as_list() -> list[dict]:
    return [{"code": c, "name": n, "label": label} for c, n, label in LANGUAGES]

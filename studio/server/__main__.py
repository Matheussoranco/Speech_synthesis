"""Start the studio: `uv run python -m server` (add --no-browser to skip opening the UI)."""
import sys
import threading
import webbrowser

import uvicorn

from . import config


def main():
    url = f"http://{'127.0.0.1' if config.HOST in ('0.0.0.0', '::') else config.HOST}:{config.PORT}"
    if "--no-browser" not in sys.argv:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"Speech Studio em {url}")
    uvicorn.run("server.main:app", host=config.HOST, port=config.PORT, log_level="warning")


if __name__ == "__main__":
    main()

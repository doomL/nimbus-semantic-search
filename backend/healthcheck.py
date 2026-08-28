#!/usr/bin/env python3
"""HTTP GET /health for Docker HEALTHCHECK (always public, no auth required)."""

from __future__ import annotations

import sys
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8000/health"


def main() -> None:
    try:
        with urllib.request.urlopen(URL, timeout=6) as resp:
            if resp.status == 200:
                sys.exit(0)
    except (urllib.error.URLError, OSError, ValueError):
        pass
    sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""HTTP GET /health for Docker HEALTHCHECK; sends Basic auth when NIMBUS_AUTH_* are set."""

from __future__ import annotations

import base64
import os
import sys
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8000/health"


def main() -> None:
    req = urllib.request.Request(URL)
    user = os.environ.get("NIMBUS_AUTH_USER", "").strip()
    password = os.environ.get("NIMBUS_AUTH_PASSWORD", "")
    if user and password:
        token = base64.b64encode(f"{user}:{password}".encode()).decode("ascii")
        req.add_header("Authorization", f"Basic {token}")
    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            if resp.status == 200:
                sys.exit(0)
    except (urllib.error.URLError, OSError, ValueError):
        pass
    sys.exit(1)


if __name__ == "__main__":
    main()

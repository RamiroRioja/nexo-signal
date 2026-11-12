#!/usr/bin/env python3
"""Descarga todos los assets públicos del release final de QualyData."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPOSITORY = "qualydata-org/desafio-qualydata"
TAG = "set-final"
USER_AGENT = "nexo-signal-final-set-downloader"


def open_url(url: str):
    request = Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
    })
    return urlopen(request, timeout=60)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destino", type=Path, default=Path("datos/final"))
    args = parser.parse_args()

    api_url = f"https://api.github.com/repos/{REPOSITORY}/releases/tags/{TAG}"
    try:
        with open_url(api_url) as response:
            release = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, OSError, json.JSONDecodeError) as exc:
        print(f"No se pudo consultar el release público {TAG}: {exc}")
        return 1

    assets = release.get("assets", [])
    if not assets:
        print(f"El release {TAG} no contiene assets descargables.")
        return 1

    args.destino.mkdir(parents=True, exist_ok=True)
    for asset in assets:
        name = str(asset.get("name", ""))
        url = str(asset.get("browser_download_url", ""))
        if not name or name in {".", ".."} or "/" in name or "\\" in name or not url:
            print("El release contiene un asset con nombre o URL no válido.")
            return 1
        destination = args.destino / name
        temporary = args.destino / f".{name}.tmp"
        try:
            with open_url(url) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            temporary.replace(destination)
        except (HTTPError, URLError, OSError) as exc:
            print(f"No se pudo descargar {name}: {exc}")
            return 1
        print(f"Descargado: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

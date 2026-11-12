#!/usr/bin/env python3
"""Descarga todos los assets públicos del release final de QualyData."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPOSITORY = "qualydata-org/desafio-qualydata"
TAG = "set-final"
USER_AGENT = "nexo-signal-final-set-downloader"
FALLBACK_ASSETS = (
    "fuentes.csv",
    "publicaciones.csv.gz",
    "comentarios.csv.gz",
    "cuentas.csv.gz",
)


def github_token() -> str | None:
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def open_url(url: str, token: str | None = None):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    return urlopen(request, timeout=60)


def download_assets(assets: list[dict[str, object]], destination_dir: Path) -> None:
    for asset in assets:
        name = str(asset.get("name", ""))
        url = str(asset.get("browser_download_url", ""))
        if not name or name in {".", ".."} or "/" in name or "\\" in name or not url:
            raise ValueError("El release contiene un asset con nombre o URL no válido.")
        destination = destination_dir / name
        temporary = destination_dir / f".{name}.tmp"
        try:
            with open_url(url) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            temporary.replace(destination)
        except (HTTPError, URLError, OSError):
            temporary.unlink(missing_ok=True)
            raise
        print(f"Descargado: {destination}")


def download_direct_assets(destination_dir: Path) -> None:
    base_url = f"https://github.com/{REPOSITORY}/releases/download/{TAG}"
    assets = [
        {"name": name, "browser_download_url": f"{base_url}/{name}"}
        for name in FALLBACK_ASSETS
    ]
    download_assets(assets, destination_dir)


def download_with_gh(destination_dir: Path) -> bool:
    gh = shutil.which("gh")
    if not gh:
        return False
    result = subprocess.run(
        [
            gh,
            "release",
            "download",
            TAG,
            "-R",
            REPOSITORY,
            "-D",
            str(destination_dir),
            "--clobber",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        print(f"gh release download falló ({result.returncode}): {detail}")
        return False
    print(f"Descarga completada con gh en {destination_dir}")
    return True


def download_rate_limit_fallback(destination_dir: Path) -> bool:
    print("La API de GitHub limitó la consulta; intentando URLs públicas de assets.")
    try:
        download_direct_assets(destination_dir)
        return True
    except (HTTPError, URLError, OSError, ValueError) as exc:
        print(f"Falló la descarga directa de assets: {exc}")
    if download_with_gh(destination_dir):
        return True
    print("No fue posible descargar los assets por URLs públicas ni con gh.")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destino", type=Path, default=Path("datos/final"))
    args = parser.parse_args()
    args.destino.mkdir(parents=True, exist_ok=True)

    api_url = f"https://api.github.com/repos/{REPOSITORY}/releases/tags/{TAG}"
    try:
        with open_url(api_url, github_token()) as response:
            release = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code in {403, 429}:
            return 0 if download_rate_limit_fallback(args.destino) else 1
        print(f"No se pudo consultar el release {TAG}: {exc}")
        return 1
    except (URLError, OSError, json.JSONDecodeError) as exc:
        print(f"No se pudo consultar el release público {TAG}: {exc}")
        return 1

    assets = release.get("assets", [])
    if not assets:
        print(f"El release {TAG} no contiene assets descargables.")
        return 1

    try:
        download_assets(assets, args.destino)
    except HTTPError as exc:
        if exc.code in {403, 429}:
            return 0 if download_rate_limit_fallback(args.destino) else 1
        print(f"No se pudo descargar un asset: {exc}")
        return 1
    except (URLError, OSError, ValueError) as exc:
        print(f"No se pudo descargar un asset: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
set -euo pipefail

REPOSITORY="qualydata-org/desafio-qualydata"
TAG="set-final"
DESTINATION="datos/final"

mkdir -p "$DESTINATION"
if command -v gh >/dev/null 2>&1 && gh release download "$TAG" -R "$REPOSITORY" -D "$DESTINATION" --clobber; then
    exit 0
fi

echo "gh no pudo descargar el release; usando la API pública de GitHub." >&2
command -v curl >/dev/null 2>&1 || {
    echo "Se necesita curl para el fallback público." >&2
    exit 1
}
release_json=$(curl -fsSL -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/${REPOSITORY}/releases/tags/${TAG}")
assets=$(python3 -c '
import json
import sys
release = json.load(sys.stdin)
for asset in release.get("assets", []):
    name = asset["name"]
    url = asset["browser_download_url"]
    print(f"{name}\t{url}")
' <<<"$release_json")

[[ -n "$assets" ]] || {
    echo "El release ${TAG} no contiene assets descargables." >&2
    exit 1
}

while IFS=$'\t' read -r asset_name asset_url; do
    [[ -n "$asset_name" && -n "$asset_url" ]] || continue
    case "$asset_name" in
        .|..|*/*)
            echo "Nombre de asset no válido: $asset_name" >&2
            exit 1
            ;;
    esac
    curl -fL --retry 3 --retry-delay 1 "$asset_url" -o "${DESTINATION}/.${asset_name}.tmp"
    mv "${DESTINATION}/.${asset_name}.tmp" "${DESTINATION}/${asset_name}"
done <<<"$assets"

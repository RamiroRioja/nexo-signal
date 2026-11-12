#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m nexo --datos datos/practica --salida salida/practica
python evaluacion/puntuar.py --verdad datos/practica/verdad --entrega salida/practica --json salida/practica/score.json
python scripts/evaluar_generalizacion.py

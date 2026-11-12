#!/usr/bin/env python3
"""Valida los archivos de entrega contra la entrada y el formato oficial."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

OUTPUT_COLUMNS = {
    "comentarios.csv": ["comentario_id", "narrativa", "postura"],
    "narrativas.csv": ["narrativa", "titulo", "frases_clave", "fecha_origen"],
    "cuentas.csv": ["cuenta_id", "grupo"],
}
COST_KEYS = {
    "comentarios_procesados", "segundos", "tokens_ia", "usd_ia",
    "usd_por_mil_comentarios", "hardware", "almacenamiento", "notas",
}
EMPTY_NARRATIVE = {"", "ninguna"}


def locate(data_dir: Path, stem: str) -> Path:
    for suffix in (".csv.gz", ".csv"):
        path = data_dir / f"{stem}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"No se encontró {stem}.csv o {stem}.csv.gz en {data_dir}")


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def parse_time(value: str) -> bool:
    try:
        datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        return True
    except (TypeError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrega", type=Path, required=True)
    parser.add_argument("--datos", type=Path, required=True)
    args = parser.parse_args()
    errors: list[str] = []

    output: dict[str, list[dict[str, str]]] = {}
    for filename, required_columns in OUTPUT_COLUMNS.items():
        path = args.entrega / filename
        if not path.is_file():
            errors.append(f"Falta {path}")
            continue
        columns, rows = read_rows(path)
        output[filename] = rows
        if columns != required_columns:
            errors.append(f"{filename}: columnas {columns}; se esperan exactamente {required_columns}")

    cost_path = args.entrega / "costo.json"
    cost: dict = {}
    if not cost_path.is_file():
        errors.append(f"Falta {cost_path}")
    else:
        try:
            cost = json.loads(cost_path.read_text(encoding="utf-8"))
            if not isinstance(cost, dict):
                errors.append("costo.json debe contener un objeto JSON.")
                cost = {}
            else:
                missing_keys = sorted(COST_KEYS - set(cost))
                extra_keys = sorted(set(cost) - COST_KEYS)
                if missing_keys or extra_keys:
                    errors.append(
                        f"costo.json: faltan claves {missing_keys}; claves extra {extra_keys}."
                    )
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"costo.json inválido: {exc}")

    try:
        _, input_comments = read_rows(locate(args.datos, "comentarios"))
        _, input_accounts = read_rows(locate(args.datos, "cuentas"))
    except (FileNotFoundError, OSError, csv.Error) as exc:
        errors.append(f"No se pudo leer la entrada: {exc}")
        input_comments, input_accounts = [], []

    comment_rows = output.get("comentarios.csv", [])
    narrative_rows = output.get("narrativas.csv", [])
    account_rows = output.get("cuentas.csv", [])

    input_ids = [row.get("comentario_id", "").strip() for row in input_comments]
    output_ids = [row.get("comentario_id", "").strip() for row in comment_rows]
    input_counts, output_counts = Counter(input_ids), Counter(output_ids)
    if any(not comment_id for comment_id in input_ids):
        errors.append("La entrada contiene comentario_id vacío.")
    if any(count != 1 for count in input_counts.values()):
        errors.append("La entrada tiene comentario_id duplicados; no se puede validar una entrega única por ID.")
    if any(not comment_id or count != 1 for comment_id, count in output_counts.items()):
        errors.append("comentarios.csv tiene IDs vacíos o duplicados.")
    if input_counts != output_counts:
        missing_ids = sum((input_counts - output_counts).values())
        extra_ids = sum((output_counts - input_counts).values())
        errors.append(f"IDs de comentarios: faltan {missing_ids:,}; sobran {extra_ids:,}.")

    narrative_ids = [row.get("narrativa", "").strip() for row in narrative_rows]
    if any(not narrative_id or narrative_id.lower() in EMPTY_NARRATIVE for narrative_id in narrative_ids):
        errors.append("narrativas.csv contiene un ID de narrativa vacío o reservado para ruido.")
    if len(narrative_ids) != len(set(narrative_ids)):
        errors.append("narrativas.csv contiene IDs de narrativa duplicados.")
    known_narratives = set(narrative_ids)
    stance_counts: Counter[str] = Counter()
    noise_count = 0
    unknown_narrative_count = 0
    nonempty_noise_stance_count = 0
    invalid_stance_count = 0
    for row in comment_rows:
        narrative = row.get("narrativa", "").strip()
        stance = row.get("postura", "").strip().lower()
        is_noise = narrative.lower() in EMPTY_NARRATIVE
        noise_count += is_noise
        stance_counts[stance or "(vacía)"] += 1
        if not is_noise and narrative not in known_narratives:
            unknown_narrative_count += 1
        if is_noise and stance:
            nonempty_noise_stance_count += 1
        if not is_noise and stance not in {"apoya", "rechaza"}:
            invalid_stance_count += 1
    if unknown_narrative_count:
        errors.append(f"{unknown_narrative_count:,} comentario(s) usan narrativas desconocidas.")
    if nonempty_noise_stance_count:
        errors.append(f"{nonempty_noise_stance_count:,} comentario(s) de ruido tienen postura no vacía.")
    if invalid_stance_count:
        errors.append(f"{invalid_stance_count:,} comentario(s) narrativos tienen postura inválida.")

    invalid_origins = [
        row.get("narrativa", "")
        for row in narrative_rows
        if not parse_time(row.get("fecha_origen", ""))
    ]
    if invalid_origins:
        errors.append(f"Fecha de origen vacía/no parseable en {len(invalid_origins)} narrativas.")

    input_account_ids = {row.get("cuenta_id", "").strip() for row in input_accounts}
    input_account_ids |= {row.get("cuenta_id", "").strip() for row in input_comments}
    input_account_ids.discard("")
    output_account_ids = [row.get("cuenta_id", "").strip() for row in account_rows]
    output_account_set = set(output_account_ids)
    if len(output_account_ids) != len(output_account_set):
        errors.append("cuentas.csv contiene cuenta_id duplicados.")
    unknown_accounts = output_account_set - input_account_ids
    missing_accounts = input_account_ids - output_account_set
    if unknown_accounts:
        errors.append(f"cuentas.csv contiene {len(unknown_accounts):,} cuenta(s) ausente(s) de la entrada.")
    if missing_accounts:
        errors.append(f"cuentas.csv no incluye {len(missing_accounts):,} cuenta(s) de la entrada.")

    group_sizes = Counter(
        row.get("grupo", "").strip()
        for row in account_rows
        if row.get("grupo", "").strip()
    )
    if cost and cost.get("comentarios_procesados") != len(input_comments):
        errors.append(
            f"costo.json indica {cost.get('comentarios_procesados')} comentarios; "
            f"la entrada tiene {len(input_comments)}."
        )

    noise_share = noise_count / len(comment_rows) if comment_rows else 0.0
    print(f"Entrega: {args.entrega}")
    print(
        f"Comentarios: {len(comment_rows):,}; narrativas: {len(narrative_rows):,}; "
        f"ruido: {noise_count:,} ({noise_share:.1%})"
    )
    print(
        f"Posturas: apoya={stance_counts['apoya']:,}; rechaza={stance_counts['rechaza']:,}; "
        f"vacías/otras={sum(count for label, count in stance_counts.items() if label not in {'apoya', 'rechaza'}):,}"
    )
    print(
        f"Cuentas: {len(account_rows):,}; grupos: {len(group_sizes):,}; "
        f"tamaños={dict(sorted(group_sizes.items()))}"
    )
    if errors:
        print(f"ERRORES ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("Entrega válida.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

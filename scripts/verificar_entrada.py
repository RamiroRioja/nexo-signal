#!/usr/bin/env python3
"""Resume la calidad de los archivos de entrada sin bloquear por datos sucios."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

EXPECTED = {
    "fuentes": {"fuente_id", "nombre", "tipo"},
    "publicaciones": {"publicacion_id", "fuente_id", "fecha", "titular"},
    "cuentas": {"cuenta_id", "usuario", "fecha_creacion", "seguidores", "siguiendo"},
    "comentarios": {
        "comentario_id", "publicacion_id", "cuenta_id", "fecha", "texto",
        "respuesta_a", "me_gusta",
    },
}
ID_COLUMNS = {
    "fuentes": "fuente_id",
    "publicaciones": "publicacion_id",
    "cuentas": "cuenta_id",
    "comentarios": "comentario_id",
}


def locate(data_dir: Path, stem: str) -> Path | None:
    for suffix in (".csv.gz", ".csv"):
        path = data_dir / f"{stem}{suffix}"
        if path.is_file():
            return path
    return None


def read_input(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def clean_ids(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].astype(str).str.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", type=Path, required=True)
    args = parser.parse_args()

    frames: dict[str, pd.DataFrame] = {}
    missing_schema = False
    print(f"Entrada: {args.datos}")
    for stem, required in EXPECTED.items():
        path = locate(args.datos, stem)
        if path is None:
            print(f"{stem}: FALTA archivo {stem}.csv o {stem}.csv.gz")
            missing_schema = True
            continue
        try:
            frame = read_input(path)
        except Exception as exc:
            print(f"{stem}: no se pudo leer {path.name}: {exc}")
            continue
        frames[stem] = frame
        missing = sorted(required - set(frame.columns))
        print(f"{stem}: {len(frame):,} filas ({path.name})")
        if missing:
            print(f"  FALTAN columnas: {', '.join(missing)}")
            missing_schema = True
            continue
        id_column = ID_COLUMNS[stem]
        ids = clean_ids(frame, id_column)
        print(f"  {id_column}: vacíos={int(ids.eq('').sum()):,}; duplicados={int(ids[ids.ne('')].duplicated().sum()):,}")

    date_columns = {
        "publicaciones": ("fecha",),
        "cuentas": ("fecha_creacion",),
        "comentarios": ("fecha",),
    }
    for stem, columns in date_columns.items():
        frame = frames.get(stem)
        if frame is None:
            continue
        for column in columns:
            if column not in frame:
                continue
            values = frame[column].astype(str).str.strip()
            parsed = pd.to_datetime(values, errors="coerce", utc=True)
            empty = int(values.eq("").sum())
            malformed = int((values.ne("") & parsed.isna()).sum())
            print(f"  {stem}.{column}: vacías={empty:,}; no parseables={malformed:,}")

    comments = frames.get("comentarios")
    if comments is not None:
        comment_ids = set(clean_ids(comments, "comentario_id")) - {""} if "comentario_id" in comments else set()
        if "respuesta_a" in comments:
            parents = comments["respuesta_a"].astype(str).str.strip()
            replies = parents.ne("")
            missing_parents = int((replies & ~parents.isin(comment_ids)).sum())
            print(f"Comentarios con respuesta_a a ID ausente: {missing_parents:,}")
        for column, target in (("publicacion_id", "publicaciones"), ("cuenta_id", "cuentas")):
            referenced = clean_ids(comments, column) if column in comments else pd.Series(dtype=str)
            target_frame = frames.get(target)
            target_column = ID_COLUMNS[target]
            known = (
                set(clean_ids(target_frame, target_column)) - {""}
                if target_frame is not None and target_column in target_frame
                else set()
            )
            unknown = int((referenced.ne("") & ~referenced.isin(known)).sum()) if target_frame is not None else 0
            print(f"Comentarios con {column} desconocido: {unknown:,}")

    return 1 if missing_schema else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Comprueba casos límite de entrada con una corrida local pequeña."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nexo.pipeline import clean_original_body


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    title_body = clean_original_body(
        "A ver, La reunión salió bien. Despierten 😡", {("a", "ver")}, {("despierten",)}
    )
    assert title_body == "La reunión salió bien", "Los títulos deben conservar texto original sin fillers ni emojis."
    with tempfile.TemporaryDirectory(prefix="nexo-robustez-") as temp:
        base = Path(temp)
        data_dir, output_dir = base / "datos", base / "salida"
        comments = [
            {
                "comentario_id": f"c{idx}", "publicacion_id": f"p{idx % 4}",
                "cuenta_id": f"u{idx}", "fecha": f"2024-01-{idx + 1:02d}T12:00:00",
                "texto": text, "respuesta_a": "", "me_gusta": "0",
            }
            for idx, text in enumerate([
                "El equipo revisó el informe durante la reunión de esta mañana.",
                "Los datos publicados muestran cambios importantes en el último período.",
                "El comunicado confirma que las tareas comenzaron según el cronograma.",
                "Varias personas solicitaron detalles adicionales sobre la decisión.",
                "El documento actualizado incluye cifras para comparar resultados.",
                "Una nueva revisión propone modificar el procedimiento utilizado.",
                "Los responsables explicaron los pasos previstos para la siguiente etapa.",
                "El registro muestra actividad estable durante varias semanas consecutivas.",
            ], start=1)
        ]
        comments.extend([
            {
                "comentario_id": "reply-short", "publicacion_id": "p1", "cuenta_id": "u9",
                "fecha": "2024-01-10T12:00:00", "texto": "Tal cual",
                "respuesta_a": "c1", "me_gusta": "0",
            },
            {
                "comentario_id": "reply-substantive", "publicacion_id": "p1", "cuenta_id": "u10",
                "fecha": "2024-01-11T12:00:00",
                "texto": "Hay otros datos que conviene revisar con más detalle.",
                "respuesta_a": "c1", "me_gusta": "0",
            },
            {
                "comentario_id": "missing-parent", "publicacion_id": "p2", "cuenta_id": "u11",
                "fecha": "2024-01-12T12:00:00",
                "texto": "La explicación presenta varios puntos que requieren una revisión.",
                "respuesta_a": "not-present", "me_gusta": "0",
            },
            {
                "comentario_id": "cycle-a", "publicacion_id": "p3", "cuenta_id": "u12",
                "fecha": "2024-01-13T12:00:00",
                "texto": "El resumen incluye información diferente para analizar.",
                "respuesta_a": "cycle-b", "me_gusta": "0",
            },
            {
                "comentario_id": "cycle-b", "publicacion_id": "p3", "cuenta_id": "u13",
                "fecha": "2024-01-14T12:00:00",
                "texto": "El reporte presenta elementos nuevos para evaluar.",
                "respuesta_a": "cycle-a", "me_gusta": "0",
            },
            {
                "comentario_id": "duplicate", "publicacion_id": "p0", "cuenta_id": "u14",
                "fecha": "2024-01-15T12:00:00",
                "texto": "La propuesta incorpora una secuencia distinta de pasos.",
                "respuesta_a": "", "me_gusta": "0",
            },
            {
                "comentario_id": "duplicate", "publicacion_id": "p0", "cuenta_id": "u15",
                "fecha": "2024-01-16T12:00:00", "texto": "",
                "respuesta_a": "", "me_gusta": "0",
            },
            {
                "comentario_id": "empty-text", "publicacion_id": "p0", "cuenta_id": "u16",
                "fecha": "2024-01-17T12:00:00", "texto": "",
                "respuesta_a": "", "me_gusta": "0",
            },
            {
                "comentario_id": "bad-date", "publicacion_id": "p0", "cuenta_id": "u17",
                "fecha": "not-a-date",
                "texto": "La revisión final contiene datos para comprobar el resultado.",
                "respuesta_a": "", "me_gusta": "0",
            },
        ])
        write_csv(data_dir / "comentarios.csv", list(comments[0]), comments)
        accounts = [
            {"cuenta_id": "account-only", "usuario": "extra", "fecha_creacion": "",
             "seguidores": "0", "siguiendo": "0"},
            {"cuenta_id": "u1", "usuario": "known", "fecha_creacion": "",
             "seguidores": "1", "siguiendo": "1"},
        ]
        write_csv(data_dir / "cuentas.csv", list(accounts[0]), accounts)
        posts = [
            {"publicacion_id": f"p{idx}", "fuente_id": "s", "fecha": "",
             "titular": ""}
            for idx in range(4)
        ]
        write_csv(data_dir / "publicaciones.csv", list(posts[0]), posts)

        subprocess.run(
            [sys.executable, "-m", "nexo", "--datos", str(data_dir),
             "--salida", str(output_dir), "--embedding", "char"],
            cwd=root, check=True,
        )

        expected_ids = Counter(row["comentario_id"] for row in comments)
        actual_ids = Counter(row["comentario_id"] for row in read_csv(output_dir / "comentarios.csv"))
        assert actual_ids == expected_ids, "La entrega debe conservar cada comentario, incluidos IDs duplicados."
        expected_accounts = {row["cuenta_id"] for row in comments + accounts}
        actual_accounts = {row["cuenta_id"] for row in read_csv(output_dir / "cuentas.csv")}
        assert actual_accounts == expected_accounts, "La entrega debe incluir la unión de cuentas."
        timings = json.loads((output_dir / "tiempos.json").read_text(encoding="utf-8"))
        assert timings and all(isinstance(value, (int, float)) for value in timings.values())
        assert all(name[0].isupper() for name in timings), "Las etapas deben tener nombres en español."

        con = duckdb.connect(str(output_dir / "nexo.duckdb"), read_only=True)
        try:
            columns = {row[0] for row in con.execute("DESCRIBE comments").fetchall()}
            assert {"narrativa", "postura", "is_coordinated", "group"} <= columns
            invalid_dates = con.execute(
                "SELECT count(*) FROM comments WHERE comentario_id = 'bad-date' AND ts IS NULL"
            ).fetchone()[0]
            assert invalid_dates == 1, "Una fecha malformada debe conservar la fila con timestamp nulo."
        finally:
            con.close()
    print("Robustez verificada: textos vacíos, fechas inválidas, padres ausentes, ciclos, IDs duplicados y unión de cuentas.")


if __name__ == "__main__":
    main()

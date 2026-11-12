#!/usr/bin/env python3
"""Crea subconjuntos por tema y período, y evalúa sin cambiar parámetros."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str], compressed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if compressed else open
    options = {"encoding": "utf-8", "newline": ""}
    with opener(path, "wt", **options) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_date(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def make_subset(
    name: str,
    selected_ids: set[str],
    comments: list[dict[str, str]],
    accounts: list[dict[str, str]],
    posts: list[dict[str, str]],
    truth_comments: list[dict[str, str]],
    truth_accounts: list[dict[str, str]],
    truth_narratives: list[dict[str, str]],
    truth_campaigns: list[dict[str, str]],
    root: Path,
) -> tuple[Path, Path]:
    base = root / "salida" / "generalizacion" / name
    data_dir, truth_dir = base / "datos", base / "verdad"
    chosen_comments = [row for row in comments if row["comentario_id"] in selected_ids]
    chosen_truth = [row for row in truth_comments if row["comentario_id"] in selected_ids]
    account_ids = {row["cuenta_id"] for row in chosen_comments}
    post_ids = {row["publicacion_id"] for row in chosen_comments}
    narrative_ids = {row["narrativa_id"] for row in chosen_truth if row["narrativa_id"] != "ninguna"}
    chosen_accounts = [row for row in accounts if row["cuenta_id"] in account_ids]
    chosen_posts = [row for row in posts if row["publicacion_id"] in post_ids]
    chosen_account_truth = [row for row in truth_accounts if row["cuenta_id"] in account_ids]
    campaign_ids = {row["campana_id"] for row in chosen_account_truth if row["campana_id"]}
    write_csv(data_dir / "comentarios.csv", chosen_comments, list(comments[0]), compressed=False)
    write_csv(data_dir / "cuentas.csv", chosen_accounts, list(accounts[0]), compressed=False)
    write_csv(data_dir / "publicaciones.csv", chosen_posts, list(posts[0]), compressed=False)
    write_csv(truth_dir / "comentarios.csv.gz", chosen_truth, list(truth_comments[0]), compressed=True)
    write_csv(truth_dir / "cuentas.csv.gz", chosen_account_truth, list(truth_accounts[0]), compressed=True)
    write_csv(
        truth_dir / "narrativas.csv",
        [row for row in truth_narratives if row["narrativa_id"] in narrative_ids],
        list(truth_narratives[0]),
    )
    write_csv(
        truth_dir / "campanas.csv",
        [row for row in truth_campaigns if row["campana_id"] in campaign_ids],
        list(truth_campaigns[0]),
    )
    return data_dir, truth_dir


def run_and_score(
    name: str, data_dir: Path, truth_dir: Path, root: Path
) -> dict:
    output_dir = root / "salida" / "generalizacion" / name / "entrega"
    subprocess.run(
        [sys.executable, "-m", "nexo", "--datos", str(data_dir), "--salida", str(output_dir)],
        cwd=root, check=True,
    )
    score_path = output_dir / "score.json"
    subprocess.run(
        [sys.executable, "evaluacion/puntuar.py", "--verdad", str(truth_dir),
         "--entrega", str(output_dir), "--json", str(score_path)],
        cwd=root, check=True,
    )
    return json.loads(score_path.read_text(encoding="utf-8"))


def print_table(rows: list[dict]) -> None:
    print("\n| Subconjunto | B-cubed F1 | Ruido F1 | Postura macro-F1 | Origen | Narrativas /30 | Coordinación F1 | Grupos B-cubed | Coord. /25 | Total /55 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        n, c = row["narrativas"], row["coordinacion"]
        print(
            f"| {row['nombre']} | {n['bcubed_f1']:.3f} | {n['ruido_f1']:.3f} | "
            f"{n['postura_macro_f1']:.3f} | {n['origen']:.3f} | {n['puntos']:.2f} | "
            f"{c['f1']:.3f} | {c['grupos_bcubed_f1']:.3f} | {c['puntos']:.2f} | "
            f"{n['puntos'] + c['puntos']:.2f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    data_dir, truth_dir = root / "datos" / "practica", root / "datos" / "practica" / "verdad"
    comments = read_csv(data_dir / "comentarios.csv.gz")
    accounts = read_csv(data_dir / "cuentas.csv.gz")
    posts = read_csv(data_dir / "publicaciones.csv.gz")
    truth_comments = read_csv(truth_dir / "comentarios.csv.gz")
    truth_accounts = read_csv(truth_dir / "cuentas.csv.gz")
    truth_narratives = read_csv(truth_dir / "narrativas.csv")
    truth_campaigns = read_csv(truth_dir / "campanas.csv")
    truth_by_id = {row["comentario_id"]: row for row in truth_comments}
    topics_by_post: dict[str, Counter[str]] = {}
    for row in comments:
        truth = truth_by_id.get(row["comentario_id"])
        if not truth or truth["narrativa_id"] == "ninguna":
            continue
        topic = truth["narrativa_id"][:1]
        topics_by_post.setdefault(row["publicacion_id"], Counter())[topic] += 1
    majority_topic = {
        post_id: min(counts, key=lambda topic: (-counts[topic], topic))
        for post_id, counts in topics_by_post.items()
    }
    selected_by_topic = {
        topic: {row["comentario_id"] for row in comments if majority_topic.get(row["publicacion_id"]) == topic}
        for topic in sorted(set(majority_topic.values()))
    }
    dates = [(row["comentario_id"], parse_date(row["fecha"])) for row in comments]
    valid_dates = [date for _, date in dates if date is not None]
    cutoff = max(valid_dates) - timedelta(days=10) if valid_dates else datetime.min
    selected_by_topic["ultimos_10_dias"] = {
        comment_id for comment_id, date in dates if date is not None and date >= cutoff
    }
    full_score_path = root / "salida" / "practica" / "score.json"
    if not full_score_path.exists():
        subprocess.run(
            [sys.executable, "evaluacion/puntuar.py", "--verdad", str(truth_dir),
             "--entrega", str(root / "salida" / "practica"), "--json", str(full_score_path)],
            cwd=root, check=True,
        )
    results = [{"nombre": "practica_completa", **json.loads(full_score_path.read_text(encoding="utf-8"))}]
    for name, ids in selected_by_topic.items():
        label = f"tema_{name}" if len(name) == 1 else name
        subset_data, subset_truth = make_subset(
            label, ids, comments, accounts, posts, truth_comments, truth_accounts,
            truth_narratives, truth_campaigns, root,
        )
        print(f"\n=== {label}: {len(ids):,} comentarios; parámetros sin cambios ===", flush=True)
        score = run_and_score(label, subset_data, subset_truth, root)
        results.append({"nombre": label, **score})
    report = root / "salida" / "generalizacion" / "score_table.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print_table(results)
    print(f"\nDetalle: {report}")


if __name__ == "__main__":
    main()

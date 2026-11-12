#!/usr/bin/env python3
"""Evalúa mezcla, coloración y orientación de postura por narrativa."""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nexo.pipeline import assign_stances, make_features, microcluster


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def diagnose(name: str, output: Path, truth_dir: Path, model_root: Path) -> list[dict]:
    truth_comments = read_csv(truth_dir / "comentarios.csv.gz")
    truth_by_id = {row["comentario_id"]: row for row in truth_comments}
    with duckdb.connect(str(output / "nexo.duckdb"), read_only=True) as con:
        frame = con.execute("SELECT * FROM comments").df()

    labels = frame["narrativa"].replace("ninguna", "").fillna("").astype(str).tolist()
    truth_ids = [truth_by_id.get(str(comment_id), {}).get("narrativa_id", "ninguna")
                 for comment_id in frame["comentario_id"]]
    truth_stances = [truth_by_id.get(str(comment_id), {}).get("postura", "")
                     for comment_id in frame["comentario_id"]]
    narrative_overlap: dict[str, Counter[str]] = defaultdict(Counter)
    for predicted, truth_id in zip(labels, truth_ids):
        if predicted and truth_id and truth_id != "ninguna":
            narrative_overlap[predicted][truth_id] += 1
    predicted_to_truth = {
        predicted: min(counts, key=lambda value: (-counts[value], value))
        for predicted, counts in narrative_overlap.items()
    }

    token_count = frame["norm"].fillna("").str.split().str.len()
    short_reply = frame["is_reply"] & (token_count <= 4) & (frame["norm"].fillna("").str.len() <= 30)
    embedded_texts = sorted(set(
        frame.loc[~frame["is_noise"] & ~short_reply, "norm"].dropna().astype(str)
    ) - {""})
    features, _ = make_features(embedded_texts, "auto", model_root)
    cluster_labels, _ = microcluster(features)
    text_to_cluster = dict(zip(embedded_texts, cluster_labels))
    micro_labels = np.asarray([int(text_to_cluster.get(text, 0)) for text in frame["norm"].fillna("")])
    cluster_stances = assign_stances(frame, micro_labels, labels)

    results = []
    for predicted, truth_id in sorted(predicted_to_truth.items(), key=lambda item: item[1]):
        indexes = [
            idx for idx, (pred, true, stance) in enumerate(zip(labels, truth_ids, truth_stances))
            if pred == predicted and true == truth_id and stance in {"apoya", "rechaza"}
            and not bool(frame.iloc[idx]["is_reply"])
        ]
        if not indexes:
            continue
        by_cluster: dict[int, Counter[str]] = defaultdict(Counter)
        for idx in indexes:
            by_cluster[int(micro_labels[idx])][truth_stances[idx]] += 1
        purity = sum(max(counts.values()) for counts in by_cluster.values()) / len(indexes)

        raw_correct = flipped_correct = final_correct = 0
        for idx in indexes:
            predicted_stance = cluster_stances.get(int(micro_labels[idx]), "apoya")
            true_stance = truth_stances[idx]
            raw_correct += predicted_stance == true_stance
            flipped = "rechaza" if predicted_stance == "apoya" else "apoya"
            flipped_correct += flipped == true_stance
            final_correct += frame.iloc[idx]["postura"] == true_stance
        raw_accuracy = raw_correct / len(indexes)
        flipped_accuracy = flipped_correct / len(indexes)
        results.append({
            "corrida": name,
            "verdad": truth_id,
            "prediccion": predicted,
            "comentarios": len(indexes),
            "microclusters": len(by_cluster),
            "pureza_micro": purity,
            "coloracion": raw_accuracy,
            "coloracion_flip": flipped_accuracy,
            "coloracion_hasta_flip": max(raw_accuracy, flipped_accuracy),
            "orientacion_invertida": "sí" if flipped_accuracy > raw_accuracy else (
                "empate" if flipped_accuracy == raw_accuracy else "no"
            ),
            "postura_final": final_correct / len(indexes),
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    model_root = root / "modelos"
    runs = [
        ("completa", root / "salida/practica", root / "datos/practica/verdad"),
        *[
            (name, root / f"salida/generalizacion/tema_{name}/entrega",
             root / f"salida/generalizacion/tema_{name}/verdad")
            for name in ("B", "H", "L")
        ],
    ]
    rows = []
    for name, output, truth in runs:
        rows.extend(diagnose(name, output, truth, model_root))

    columns = [
        "corrida", "verdad", "prediccion", "comentarios", "microclusters",
        "pureza_micro", "coloracion", "coloracion_flip", "coloracion_hasta_flip",
        "orientacion_invertida", "postura_final",
    ]
    print("| " + " | ".join(columns) + " |")
    print("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        print("| " + " | ".join(
            f"{row[column]:.3f}" if isinstance(row[column], float) else str(row[column])
            for column in columns
        ) + " |")
    report = root / "salida/diagnostico_postura.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Detalle: {report}")


if __name__ == "__main__":
    main()

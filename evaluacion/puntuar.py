"""Puntaje automático del Desafío QualyData (narrativas y coordinación).

Compara la entrega de un equipo con la verdad del set y calcula la parte automática
del puntaje (55 de 100 puntos). El resto lo evalúa el jurado.

Uso:
    python evaluacion/puntuar.py --verdad datos/practica/verdad --entrega salida/practica
    python evaluacion/puntuar.py --verdad datos/practica/verdad --entrega salida/practica --json resultado.json

Stdlib only so it runs anywhere without installing anything.
"""
import argparse
import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

EMPTY = {"", "ninguna", "none", "null"}


def open_csv(path: Path) -> List[dict]:
    if not path.exists():
        gz = path.with_suffix(path.suffix + ".gz")
        if gz.exists():
            path = gz
        else:
            sys.exit(f"Falta el archivo {path}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def require(rows: List[dict], columns: List[str], name: str):
    if rows and not set(columns) <= set(rows[0]):
        sys.exit(f"{name} debe tener las columnas {columns}; tiene {list(rows[0])}")


def bcubed(truth: Dict[str, str], pred: Dict[str, str]) -> Tuple[float, float, float]:
    """B-cubed precision/recall/F1 over the items in `truth`.

    An item without a predicted cluster counts as its own singleton cluster.
    """
    pairs = Counter()
    size_t = Counter()
    size_p = Counter()
    for item, t in truth.items():
        p = pred.get(item, "")
        p = f"__solo__{item}" if p.strip().lower() in EMPTY else p
        pairs[(p, t)] += 1
        size_t[t] += 1
        size_p[p] += 1
    n = len(truth)
    if n == 0:
        return 0.0, 0.0, 0.0
    precision = sum(c * c / size_p[p] for (p, _), c in pairs.items()) / n
    recall = sum(c * c / size_t[t] for (_, t), c in pairs.items()) / n
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def f1_binary(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def parse_time(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value.strip())
    except (ValueError, AttributeError):
        return None


def score_narratives(truth_dir: Path, sub_dir: Path) -> dict:
    t_rows = open_csv(truth_dir / "comentarios.csv")
    s_rows = open_csv(sub_dir / "comentarios.csv")
    require(s_rows, ["comentario_id", "narrativa", "postura"], "entrega/comentarios.csv")
    pred_label = {r["comentario_id"]: (r.get("narrativa") or "").strip() for r in s_rows}
    pred_stance = {r["comentario_id"]: (r.get("postura") or "").strip().lower() for r in s_rows}

    truth_narr = {r["comentario_id"]: r["narrativa_id"] for r in t_rows if r["narrativa_id"] != "ninguna"}
    _, _, b_f1 = bcubed(truth_narr, pred_label)
    b_p, b_r, _ = bcubed(truth_narr, pred_label)

    # Noise detection: F1 on the "ninguna" class.
    tp = fp = fn = 0
    for r in t_rows:
        is_noise = r["narrativa_id"] == "ninguna"
        said_noise = pred_label.get(r["comentario_id"], "").lower() in EMPTY
        tp += is_noise and said_noise
        fp += (not is_noise) and said_noise
        fn += is_noise and not said_noise
    _, _, noise_f1 = f1_binary(tp, fp, fn)

    # Stance: macro-F1 over apoya/rechaza on true narrative comments.
    stance_f1 = []
    for cls in ("apoya", "rechaza"):
        tp = fp = fn = 0
        for r in t_rows:
            if r["narrativa_id"] == "ninguna":
                continue
            truth_is = r["postura"] == cls
            pred_is = pred_stance.get(r["comentario_id"], "") == cls
            tp += truth_is and pred_is
            fp += (not truth_is) and pred_is
            fn += truth_is and not pred_is
        stance_f1.append(f1_binary(tp, fp, fn)[2])
    stance = sum(stance_f1) / len(stance_f1)

    # Origin of emergent narratives: match each one to the team's cluster with the largest overlap.
    narr_truth = open_csv(truth_dir / "narrativas.csv")
    sub_narr_path = sub_dir / "narrativas.csv"
    sub_narr = open_csv(sub_narr_path) if sub_narr_path.exists() else []
    origin_by_label = {r.get("narrativa", "").strip(): parse_time(r.get("fecha_origen", "")) for r in sub_narr}
    overlap = defaultdict(Counter)
    for cid, t in truth_narr.items():
        p = pred_label.get(cid, "")
        if p.lower() not in EMPTY:
            overlap[t][p] += 1
    origin_scores = {}
    for n in narr_truth:
        if n["emergente"] != "si":
            continue
        true_t = parse_time(n["fecha_origen"])
        best = overlap[n["narrativa_id"]].most_common(1)
        guess = origin_by_label.get(best[0][0]) if best else None
        if not guess or not true_t:
            origin_scores[n["narrativa_id"]] = 0.0
            continue
        err_h = abs((guess - true_t).total_seconds()) / 3600
        origin_scores[n["narrativa_id"]] = 1.0 if err_h <= 3 else max(0.0, 1 - (err_h - 3) / 45)
    origin = sum(origin_scores.values()) / len(origin_scores) if origin_scores else 0.0

    combined = 0.5 * b_f1 + 0.15 * noise_f1 + 0.2 * stance + 0.15 * origin
    return {"bcubed_precision": b_p, "bcubed_recall": b_r, "bcubed_f1": b_f1, "ruido_f1": noise_f1,
            "postura_macro_f1": stance, "origen": origin, "origen_por_narrativa": origin_scores,
            "combinado": combined, "puntos": round(30 * combined, 2)}


def score_coordination(truth_dir: Path, sub_dir: Path) -> dict:
    t_rows = open_csv(truth_dir / "cuentas.csv")
    s_rows = open_csv(sub_dir / "cuentas.csv")
    require(s_rows, ["cuenta_id", "grupo"], "entrega/cuentas.csv")
    pred = {r["cuenta_id"]: (r.get("grupo") or "").strip() for r in s_rows}
    tp = fp = fn = 0
    for r in t_rows:
        coordinated = bool(r["campana_id"])
        flagged = pred.get(r["cuenta_id"], "").lower() not in EMPTY
        tp += coordinated and flagged
        fp += (not coordinated) and flagged
        fn += coordinated and not flagged
    p, r, f1 = f1_binary(tp, fp, fn)
    truth_groups = {row["cuenta_id"]: row["campana_id"] for row in t_rows if row["campana_id"]}
    _, _, groups = bcubed(truth_groups, pred)
    combined = 0.7 * f1 + 0.3 * groups
    return {"precision": p, "recall": r, "f1": f1, "cuentas_marcadas": tp + fp,
            "grupos_bcubed_f1": groups, "combinado": combined, "puntos": round(25 * combined, 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verdad", required=True, type=Path)
    ap.add_argument("--entrega", required=True, type=Path)
    ap.add_argument("--json", type=Path, help="Guardar el resultado completo en un archivo JSON")
    args = ap.parse_args()

    narr = score_narratives(args.verdad, args.entrega)
    coord = score_coordination(args.verdad, args.entrega)
    cost_path = args.entrega / "costo.json"
    cost = json.loads(cost_path.read_text(encoding="utf-8")) if cost_path.exists() else None

    print("NARRATIVAS (hasta 30 puntos)")
    print(f"  Agrupamiento B-cubed  P={narr['bcubed_precision']:.3f}  R={narr['bcubed_recall']:.3f}  F1={narr['bcubed_f1']:.3f}")
    print(f"  Detección de ruido    F1={narr['ruido_f1']:.3f}")
    print(f"  Postura               macro-F1={narr['postura_macro_f1']:.3f}")
    print(f"  Origen de emergentes  {narr['origen']:.3f}  {narr['origen_por_narrativa']}")
    print(f"  -> {narr['puntos']} puntos")
    print("COORDINACIÓN (hasta 25 puntos)")
    print(f"  Cuentas coordinadas   P={coord['precision']:.3f}  R={coord['recall']:.3f}  F1={coord['f1']:.3f}  (marcadas: {coord['cuentas_marcadas']})")
    print(f"  Grupos B-cubed        F1={coord['grupos_bcubed_f1']:.3f}")
    print(f"  -> {coord['puntos']} puntos")
    print(f"TOTAL AUTOMÁTICO: {round(narr['puntos'] + coord['puntos'], 2)} / 55")
    if cost is None:
        print("Aviso: falta entrega/costo.json (lo revisa el jurado para el criterio de costo y escala).")
    if args.json:
        args.json.write_text(json.dumps({"narrativas": narr, "coordinacion": coord, "costo": cost},
                                        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

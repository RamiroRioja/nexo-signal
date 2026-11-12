"""Línea base mínima del Desafío QualyData.

Es un punto de partida, no una solución: agrupa comentarios con TF-IDF + k-means,
asigna postura con una lista de palabras y marca cuentas que comentan juntas en
ventanas cortas. Sirve para entender los datos, el formato de entrega y el puntaje.

Uso:
    python ejemplos/linea_base.py --datos datos/practica --entrega salida/practica
    python evaluacion/puntuar.py --verdad datos/practica/verdad --entrega salida/practica
"""
import argparse
import json
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

ABBREVIATIONS = {"q": "que", "xq": "porque", "x": "por", "tmb": "tambien", "pa": "para", "tdo": "todo"}
STOPWORDS = set("""a al algo con como de del el ella en es esa ese eso esta este esto la las le lo los
mas me mi muy no nos o para pero por que se si sin su sus te todo todos un una uno y ya ahi aca hay
""".split())
REJECT_MARKERS = ["mentira", "verso", "falso", "fake", "nada que ver", "no es cierto", "trucho",
                  "desmintio", "sin pruebas", "inventan", "dejen de", "no hubo", "nadie va a"]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"@\w+|#\w+|https?://\S+", " ", text)
    text = re.sub(r"[^a-zñ0-9\s]", " ", text)
    text = re.sub(r"(\w)\1{2,}", r"\1", text)
    return " ".join(ABBREVIATIONS.get(w, w) for w in text.split())


def cluster_narratives(df: pd.DataFrame, k: int) -> pd.Series:
    content = df["norm"].str.split().str.len() >= 4
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=20, max_df=0.4,
                          stop_words=list(STOPWORDS), sublinear_tf=True)
    X = vec.fit_transform(df.loc[content, "norm"])
    km = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=4096, n_init=3)
    labels = km.fit_predict(X)
    out = pd.Series("ninguna", index=df.index)
    out.loc[content] = [f"k{l:02d}" for l in labels]
    terms = np.array(vec.get_feature_names_out())
    top = {f"k{i:02d}": list(terms[np.argsort(-c)[:6]]) for i, c in enumerate(km.cluster_centers_)}
    return out, top


def stance(norm_text: str) -> str:
    return "rechaza" if any(m in norm_text for m in REJECT_MARKERS) else "apoya"


def coordinated_groups(df: pd.DataFrame, window_s: int = 1800, min_accounts: int = 3, min_tokens: int = 6,
                       min_events: int = 2, min_size: int = 4) -> dict:
    """Link accounts that post the same long text on the same post within a short window.

    Copy-paste bursts are a classic coordination signal. This misses campaigns that
    vary their wording, which is exactly what a real solution has to catch.
    """
    long_texts = df[df["norm"].str.split().str.len() >= min_tokens]
    links = defaultdict(int)
    for (_, _), g in long_texts.groupby(["publicacion_id", "norm"]):
        if g["cuenta_id"].nunique() < min_accounts:
            continue
        g = g.sort_values("ts")
        ts, acc = g["ts"].to_numpy(), g["cuenta_id"].to_numpy()
        burst = [a for a, t in zip(acc, ts) if t - ts[0] <= window_s]
        burst = sorted(set(burst))
        if len(burst) < min_accounts:
            continue
        for i in range(len(burst)):
            for j in range(i + 1, len(burst)):
                links[(burst[i], burst[j])] += 1
    graph = defaultdict(set)
    for (a, b), events in links.items():
        if events >= min_events:
            graph[a].add(b)
            graph[b].add(a)
    groups, seen = {}, set()
    for node in graph:
        if node in seen:
            continue
        stack, comp = [node], []
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.append(cur)
            stack.extend(graph[cur] - seen)
        if len(comp) >= min_size:
            label = f"g{len(set(groups.values())):03d}"
            for acc in comp:
                groups[acc] = label
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", type=Path, default=Path("datos/practica"))
    ap.add_argument("--entrega", type=Path, default=Path("salida/practica"))
    ap.add_argument("--k", type=int, default=20, help="Cantidad de grupos de k-means")
    args = ap.parse_args()
    started = time.time()

    df = pd.read_csv(args.datos / "comentarios.csv.gz", dtype=str, keep_default_na=False)
    df["fecha_dt"] = pd.to_datetime(df["fecha"], utc=True)
    df["ts"] = df["fecha_dt"].astype("int64") // 10**9
    df["norm"] = df["texto"].map(normalize)

    df["narrativa"], top_terms = cluster_narratives(df, args.k)
    df["postura"] = np.where(df["narrativa"] == "ninguna", "", df["norm"].map(stance))

    args.entrega.mkdir(parents=True, exist_ok=True)
    df[["comentario_id", "narrativa", "postura"]].to_csv(args.entrega / "comentarios.csv", index=False)

    rows = []
    for label, g in df[df["narrativa"] != "ninguna"].groupby("narrativa"):
        origin = g["fecha_dt"].quantile(0.01).tz_convert("-03:00")
        rows.append({"narrativa": label, "titulo": " / ".join(top_terms[label][:3]),
                     "frases_clave": "|".join(top_terms[label]), "fecha_origen": origin.isoformat()})
    pd.DataFrame(rows).to_csv(args.entrega / "narrativas.csv", index=False)

    groups = coordinated_groups(df)
    accounts = pd.DataFrame({"cuenta_id": df["cuenta_id"].unique()})
    accounts["grupo"] = accounts["cuenta_id"].map(groups).fillna("")
    accounts.to_csv(args.entrega / "cuentas.csv", index=False)

    elapsed = round(time.time() - started, 1)
    cost = {"comentarios_procesados": len(df), "segundos": elapsed, "tokens_ia": 0, "usd_ia": 0.0,
            "usd_por_mil_comentarios": 0.0, "hardware": "notebook, CPU", "notas": "línea base sin IA"}
    (args.entrega / "costo.json").write_text(json.dumps(cost, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Listo en {elapsed}s: {df['narrativa'].nunique() - 1} grupos, {len(set(groups.values()))} grupos coordinados")


if __name__ == "__main__":
    main()

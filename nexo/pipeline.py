"""Pipeline determinista de narrativas y coordinación, ejecutado en CPU."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import timedelta
from itertools import combinations
from pathlib import Path

import duckdb
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2_normalize

SEED = 42
MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_DIRNAME = "paraphrase-multilingual-MiniLM-L12-v2"
STOPWORDS = set(
    """a al algo ante bajo cabe cada casi como con contra cual cuando de del desde donde
    durante e el ella ellas ellos en entre era es esa esas ese eso esos esta estaba estaban
    estas este esto estos fue fueron ha han hasta hay la las le les lo los mas me mi mis
    mucha muchas mucho muchos muy ni no nos o os otra otras otro otros para pero poco por
    porque que quien se sea sean segun ser si sin sobre su sus te tener tiene todo todos tras
    tu un una unas uno unos y ya""".split()
)
REFUTATION_MARKERS = (
    "mentira", "falso", "no es cierto", "nadie va a", "verso", "nada que ver",
    "sin pruebas", "no hubo", "inventan", "trucho", "fake",
)
FALLBACK_FILLERS = (
    "lo que nadie dice es que", "la verdad que", "te digo algo", "a ver", "che",
    "sinceramente", "abran los ojos", "despierten", "asi estamos", "que bronca",
    "lamentable",
)
SHORT_AGREEMENT_MAX_TOKENS = 4
SHORT_AGREEMENT_MAX_CHARS = 30
SHORT_AGREEMENTS = {
    "asi es", "bien dicho", "claro", "coincido", "correcto", "de acuerdo",
    "efectivamente", "es asi", "exacto", "lo mismo digo", "por supuesto",
    "tal cual", "totalmente", "tremendo pero cierto", "100 de acuerdo", "si",
}


def is_short_agreement(text: str) -> bool:
    if len(text.split()) > SHORT_AGREEMENT_MAX_TOKENS or len(text) > SHORT_AGREEMENT_MAX_CHARS:
        return False
    if text in SHORT_AGREEMENTS:
        return True
    return any(SequenceMatcher(None, text, phrase).ratio() >= 0.84 for phrase in SHORT_AGREEMENTS)


def log_stage(name: str, started: float) -> float:
    elapsed = time.perf_counter() - started
    print(f"[{name}] listo en {elapsed:.1f}s", flush=True)
    return elapsed


def find_input(data_dir: Path, stem: str) -> Path:
    for name in (f"{stem}.csv.gz", f"{stem}.csv"):
        path = data_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No se encontró {stem}.csv[.gz] en {data_dir}")


def sql_path(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def import_csv(con: duckdb.DuckDBPyConnection, path: Path, table: str, parquet: Path) -> None:
    con.execute(
        f"CREATE OR REPLACE TABLE {table} AS "
        f"SELECT * FROM read_csv_auto({sql_path(path)}, header=true, all_varchar=true, "
        "ignore_errors=true, strict_mode=false, union_by_name=true)"
    )
    con.execute(
        f"COPY (SELECT * FROM {table}) TO {sql_path(parquet)} "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )


def fold_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def normalize_text(value: object) -> tuple[str, str]:
    raw = "" if value is None or pd.isna(value) else str(value)
    hashtags = " ".join(fold_accents(tag) for tag in re.findall(r"(?<!\w)#([\w]+)", raw))
    text = re.sub(r"https?://\S+|www\.\S+", " ", raw, flags=re.I)
    text = re.sub(r"(?<!\w)@[\w.]+", " ", text)
    text = re.sub(r"(?<!\w)#[\w]+", " ", text)
    text = fold_accents(text)
    text = "".join(" " if unicodedata.category(char) in {"So", "Sk", "Cs"} else char for char in text)
    text = re.sub(r"[^a-z0-9ñ\s]", " ", text)
    text = re.sub(r"([a-z0-9])\1{2,}", r"\1\1", text)
    return " ".join(text.split()), hashtags


def detect_fillers(texts: list[str]) -> tuple[set[tuple[str, ...]], set[tuple[str, ...]]]:
    prefix_counts: dict[tuple[str, ...], set[str]] = defaultdict(set)
    suffix_counts: dict[tuple[str, ...], set[str]] = defaultdict(set)
    prefix_continuations: dict[tuple[str, ...], set[tuple[str, ...]]] = defaultdict(set)
    suffix_contexts: dict[tuple[str, ...], set[tuple[str, ...]]] = defaultdict(set)
    unique_texts = set(texts)
    for text in unique_texts:
        tokens = text.split()
        for size in range(2, min(5, len(tokens) - 2) + 1):
            prefix, prefix_body = tuple(tokens[:size]), tokens[size:]
            suffix, suffix_body = tuple(tokens[-size:]), tokens[:-size]
            prefix_counts[prefix].add(" ".join(prefix_body))
            prefix_continuations[prefix].add(tuple(prefix_body[:2]))
            suffix_counts[suffix].add(" ".join(suffix_body))
            suffix_contexts[suffix].add(tuple(suffix_body[-2:]))
    min_remainders = max(30, min(100, int(len(unique_texts) * 0.003)))
    min_diversity = max(8, int(math.sqrt(min_remainders)))
    protected = {"no", "nada", "sin", "mentira", "falso", "nadie", "verso", "fake", "trucho"}
    prefixes = {
        phrase for phrase, bodies in prefix_counts.items()
        if len(bodies) >= min_remainders and len(prefix_continuations[phrase]) >= min_diversity
        and phrase[0] not in protected
    }
    suffixes = {
        phrase for phrase, bodies in suffix_counts.items()
        if len(bodies) >= min_remainders and len(suffix_contexts[phrase]) >= min_diversity
        and phrase[0] not in protected
    }
    prefixes.update(tuple(fold_accents(item).split()) for item in FALLBACK_FILLERS[:6])
    suffixes.update(tuple(fold_accents(item).split()) for item in FALLBACK_FILLERS[6:])
    return prefixes, suffixes


def strip_fillers(text: str, prefixes: set[tuple[str, ...]], suffixes: set[tuple[str, ...]]) -> str:
    tokens = text.split()
    changed = True
    while changed and len(tokens) > 3:
        changed = False
        for size in range(min(5, len(tokens) - 2), 0, -1):
            if tuple(tokens[:size]) in prefixes:
                tokens = tokens[size:]
                changed = True
                break
    changed = True
    while changed and len(tokens) > 3:
        changed = False
        for size in range(min(5, len(tokens) - 2), 0, -1):
            if tuple(tokens[-size:]) in suffixes:
                tokens = tokens[:-size]
                changed = True
                break
    return " ".join(tokens)


def extract_features(text: str) -> tuple[int, int, int]:
    tokens = text.split()
    content = sum(token not in STOPWORDS and len(token) > 1 for token in tokens)
    return len(tokens), content, len(text)


def classify_noise(frame: pd.DataFrame) -> np.ndarray:
    counts, content, lengths = zip(*(extract_features(text) for text in frame["norm"])) if len(frame) else ([], [], [])
    counts_array = np.asarray(counts)
    content_array = np.asarray(content)
    lengths_array = np.asarray(lengths)
    repeated = frame.groupby("norm", sort=False).agg(
        uses=("comentario_id", "size"), posts=("publicacion_id", "nunique")
    )
    repeat_rows = frame["norm"].map(repeated["uses"]).fillna(0).to_numpy()
    repeat_posts = frame["norm"].map(repeated["posts"]).fillna(0).to_numpy()
    raw_empty = frame["norm"].eq("").to_numpy()
    short_repeat = (repeat_rows >= 8) & (repeat_posts >= 3) & (counts_array <= 7)
    noise = raw_empty | (counts_array <= 1) | ((counts_array <= 3) & (content_array <= 1))
    noise |= (counts_array <= 5) & (lengths_array <= 36) & (content_array <= 2) & (repeat_rows >= 2)
    noise |= short_repeat & (counts_array <= 4)
    question = frame["texto"].fillna("").astype(str).str.contains(r"\?", regex=True).to_numpy()
    noise |= question & (counts_array <= 5) & (content_array <= 3)
    noise |= question & (counts_array <= 7) & (repeat_rows >= 2) & (content_array <= 4)
    return noise


def ensure_model(model_root: Path):
    from sentence_transformers import SentenceTransformer

    target = model_root / MODEL_DIRNAME
    model_root.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    model = SentenceTransformer(MODEL_ID, cache_folder=str(model_root), device="cpu")
    model.save(str(target))
    return target


def load_embedding_model(model_root: Path):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    target = ensure_model(model_root)
    return SentenceTransformer(str(target), device="cpu")


def cached_embeddings(texts: list[str], model_root: Path, model, batch_size: int = 128) -> np.ndarray:
    cache_path = model_root / "embedding_cache.duckdb"
    cache = duckdb.connect(str(cache_path))
    cache.execute("CREATE TABLE IF NOT EXISTS embeddings (text_hash VARCHAR PRIMARY KEY, vector BLOB)")
    hashes = [hashlib.sha256((MODEL_ID + "\n" + text).encode("utf-8")).hexdigest() for text in texts]
    vectors: list[np.ndarray | None] = [None] * len(texts)
    missing_positions: list[int] = []
    for offset in range(0, len(hashes), 4000):
        subset = hashes[offset:offset + 4000]
        placeholders = ",".join("?" for _ in subset)
        rows = cache.execute(
            f"SELECT text_hash, vector FROM embeddings WHERE text_hash IN ({placeholders})", subset
        ).fetchall()
        found = {key: np.frombuffer(blob, dtype=np.float32) for key, blob in rows}
        for pos in range(offset, min(offset + 4000, len(hashes))):
            vector = found.get(hashes[pos])
            if vector is None:
                missing_positions.append(pos)
            else:
                vectors[pos] = vector
    if missing_positions:
        batch = [texts[pos] for pos in missing_positions]
        encoded = model.encode(
            batch, batch_size=batch_size, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True, device="cpu",
        ).astype(np.float32, copy=False)
        rows_to_insert = []
        for pos, vector in zip(missing_positions, encoded):
            vectors[pos] = vector
            rows_to_insert.append((hashes[pos], vector.tobytes()))
        cache.executemany("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", rows_to_insert)
    cache.close()
    return np.vstack(vectors).astype(np.float32) if vectors else np.empty((0, 384), dtype=np.float32)


def make_features(texts: list[str], embedding_mode: str, model_root: Path) -> tuple[np.ndarray, str]:
    if not texts:
        return np.empty((0, 0), dtype=np.float32), "sin textos"
    vectorizer_args = {
        "analyzer": "char", "ngram_range": (3, 5), "min_df": 2 if len(texts) > 1 else 1,
        "max_features": 30000, "sublinear_tf": True, "dtype": np.float32,
    }
    try:
        char = TfidfVectorizer(**vectorizer_args).fit_transform(texts)
    except ValueError:
        vectorizer_args.update(ngram_range=(1, 3), min_df=1)
        char = TfidfVectorizer(**vectorizer_args).fit_transform(texts)
    if min(char.shape) <= 2:
        char_dense = char.toarray()
    else:
        components = min(96, char.shape[0] - 1, char.shape[1] - 1)
        char_dense = TruncatedSVD(n_components=components, random_state=SEED, n_iter=5).fit_transform(char)
    char_dense = l2_normalize(char_dense).astype(np.float32, copy=False)
    if embedding_mode != "char":
        try:
            model = load_embedding_model(model_root)
            semantic = cached_embeddings(texts, model_root, model)
            features = np.hstack((semantic * 0.78, char_dense * 0.42))
            return l2_normalize(features).astype(np.float32, copy=False), "MiniLM + n-gramas de caracteres"
        except Exception as exc:
            if embedding_mode == "minilm":
                raise
            print(f"[embeddings] no disponible ({exc}); uso n-gramas de caracteres", file=sys.stderr, flush=True)
    return char_dense, "n-gramas de caracteres + SVD"


def microcluster(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(features)
    if n == 0:
        return np.empty(0, dtype=int), np.empty((0, 0), dtype=np.float32)
    k = min(n, max(12, min(300, int(math.sqrt(n) * 1.7))))
    if k >= n:
        return np.arange(n, dtype=int), features
    model = MiniBatchKMeans(
        n_clusters=k, random_state=SEED, batch_size=2048, n_init=5,
        max_iter=100, reassignment_ratio=0.01,
    )
    labels = model.fit_predict(features)
    centers = l2_normalize(model.cluster_centers_).astype(np.float32, copy=False)
    return labels, centers


def choose_narrative_communities(
    frame: pd.DataFrame, labels: np.ndarray, centers: np.ndarray, n_clusters: int
) -> dict[int, int]:
    graph = nx.Graph()
    graph.add_nodes_from(range(n_clusters))
    if n_clusters == 0:
        return {}
    similarities = centers @ centers.T
    for left in range(n_clusters):
        for right in np.argsort(-similarities[left])[1:9]:
            similarity = float(similarities[left, right])
            if similarity >= 0.43:
                graph.add_edge(left, int(right), weight=max(0.05, similarity - 0.30))
    id_to_index = {}
    for idx, comment_id in enumerate(frame["comentario_id"].astype(str)):
        id_to_index.setdefault(comment_id, idx)
    cluster_by_comment = dict(zip(frame.index, labels))
    reply_counts: Counter[tuple[int, int]] = Counter()
    for idx, row in frame.iterrows():
        parent_idx = id_to_index.get(str(row["respuesta_a"])) if row["respuesta_a"] else None
        if parent_idx is None or idx not in cluster_by_comment or parent_idx not in cluster_by_comment:
            continue
        child_cluster, parent_cluster = int(cluster_by_comment[idx]), int(cluster_by_comment[parent_idx])
        if child_cluster != parent_cluster:
            reply_counts[tuple(sorted((child_cluster, parent_cluster)))] += 1
    for (left, right), count in reply_counts.items():
        if count >= 2:
            old = graph[left][right]["weight"] if graph.has_edge(left, right) else 0.0
            graph.add_edge(left, right, weight=old + 1.5 + math.log1p(count))
    communities = nx.community.louvain_communities(graph, weight="weight", seed=SEED, resolution=0.9)
    return {int(cluster): community_id for community_id, members in enumerate(communities) for cluster in members}


def resolve_replies(frame: pd.DataFrame, initial: list[str], is_noise: np.ndarray) -> tuple[list[str], list[str]]:
    labels = list(initial)
    stances = [""] * len(frame)
    id_to_index = {}
    for idx, comment_id in enumerate(frame["comentario_id"].astype(str)):
        id_to_index.setdefault(comment_id, idx)
    parents = [id_to_index.get(parent) if parent else None for parent in frame["respuesta_a"].fillna("").astype(str)]
    for idx, parent in enumerate(parents):
        if parent is not None:
            labels[idx] = ""
    norms = frame["norm"].tolist()
    token_counts = [len(text.split()) for text in norms]
    for start in range(len(frame)):
        if parents[start] is None or labels[start]:
            continue
        path, seen, cur = [], set(), start
        while parents[cur] is not None and cur not in seen and not labels[cur]:
            seen.add(cur)
            path.append(cur)
            cur = parents[cur]
        if cur in seen:
            cycle_start = path.index(cur)
            cycle = path[cycle_start:]
            cycle_label = next((initial[node] for node in cycle if initial[node]), "")
            for node in cycle:
                labels[node] = cycle_label
            path = path[:cycle_start]
        parent_label = labels[cur]
        for node in reversed(path):
            labels[node] = parent_label or initial[node]
            stances[node] = ""
            parent_label = labels[node]
    for idx in range(len(frame)):
        if labels[idx] and parents[idx] is None:
            stances[idx] = "apoya"
        if is_noise[idx]:
            labels[idx] = ""
            stances[idx] = ""
    return labels, stances


def assign_stances(frame: pd.DataFrame, micro_labels: np.ndarray, narrative_labels: list[str]) -> dict[int, str]:
    graph = nx.Graph()
    signed_edges: Counter[tuple[int, int]] = Counter()
    id_to_index = {}
    for idx, comment_id in enumerate(frame["comentario_id"].astype(str)):
        id_to_index.setdefault(comment_id, idx)
    for idx, row in frame.iterrows():
        parent_idx = id_to_index.get(str(row["respuesta_a"])) if row["respuesta_a"] else None
        if parent_idx is None or parent_idx == idx or not narrative_labels[idx] or not narrative_labels[parent_idx]:
            continue
        if narrative_labels[idx] != narrative_labels[parent_idx]:
            continue
        text = row["norm"]
        same = is_short_agreement(text)
        if same:
            continue
        left, right = int(micro_labels[parent_idx]), int(micro_labels[idx])
        if left != right:
            pair = tuple(sorted((left, right)))
            signed_edges[pair] -= 1
    for (left, right), relation in signed_edges.items():
        if relation:
            graph.add_edge(left, right, weight=abs(relation), relation=1 if relation > 0 else -1)
    cluster_to_narrative: dict[int, str] = {}
    cluster_rows: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(narrative_labels):
        if label:
            cluster = int(micro_labels[idx])
            cluster_to_narrative.setdefault(cluster, label)
            cluster_rows[cluster].append(idx)
    cluster_stance = {}
    for narrative in sorted(set(narrative_labels) - {""}):
        clusters = [cluster for cluster, assigned in cluster_to_narrative.items() if assigned == narrative]
        if not clusters:
            continue
        earliest, marker_share = {}, {}
        for cluster in clusters:
            rows = cluster_rows[cluster]
            times = frame.iloc[rows]["ts"].dropna().tolist()
            earliest[cluster] = min(times) if times else pd.Timestamp.max.tz_localize("UTC")
            marker_share[cluster] = sum(
                any(marker in text for marker in REFUTATION_MARKERS)
                for text in frame.iloc[rows]["norm"].tolist()
            ) / max(1, len(rows))
        colors = {}
        ordered = sorted(
            clusters,
            key=lambda cluster: (-(graph.degree(cluster, weight="weight") if graph.has_node(cluster) else 0),
                                 earliest[cluster]),
        )
        for seed in ordered:
            if seed in colors:
                continue
            colors[seed] = 0
            queue = [seed]
            while queue:
                cur = queue.pop(0)
                if not graph.has_node(cur):
                    continue
                for neighbor in graph.neighbors(cur):
                    if neighbor not in clusters:
                        continue
                    expected = colors[cur] if graph[cur][neighbor]["relation"] > 0 else 1 - colors[cur]
                    if neighbor not in colors:
                        colors[neighbor] = expected
                        queue.append(neighbor)
        if len(set(colors.values())) == 1 and len(clusters) > 1:
            midpoint = sorted(earliest.values())[len(earliest) // 2]
            colors = {cluster: int(earliest[cluster] > midpoint) for cluster in clusters}
        support_cluster = min(clusters, key=lambda cluster: earliest[cluster])
        reject_cluster = max(clusters, key=lambda cluster: marker_share[cluster])
        if marker_share[reject_cluster] >= 0.20 and reject_cluster != support_cluster:
            colors[reject_cluster] = 1 - colors[support_cluster]
        for cluster in clusters:
            cluster_stance[cluster] = "apoya" if colors.get(cluster, 0) == colors[support_cluster] else "rechaza"
    return cluster_stance


def propagate_reply_stances(frame: pd.DataFrame, labels: list[str], stances: list[str]) -> list[str]:
    id_to_index = {}
    for idx, comment_id in enumerate(frame["comentario_id"].astype(str)):
        id_to_index.setdefault(comment_id, idx)
    parents = [id_to_index.get(parent) if parent else None for parent in frame["respuesta_a"].fillna("").astype(str)]
    norms = frame["norm"].tolist()
    for start in range(len(frame)):
        if parents[start] is None or not labels[start] or stances[start]:
            continue
        path, seen, cur = [], set(), start
        while parents[cur] is not None and cur not in seen and not stances[cur]:
            seen.add(cur)
            path.append(cur)
            cur = parents[cur]
        if cur in seen:
            for node in path:
                stances[node] = stances[node] or "apoya"
            continue
        parent_stance = stances[cur]
        for node in reversed(path):
            same = is_short_agreement(norms[node])
            stances[node] = parent_stance if same and parent_stance else (
                "rechaza" if parent_stance == "apoya" else "apoya"
            )
            parent_stance = stances[node]
    return stances


def narrative_outputs(frame: pd.DataFrame, labels: list[str], stances: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    assigned = frame.copy()
    assigned["narrativa"] = labels
    assigned["postura"] = stances
    assigned["hour"] = assigned["ts"].dt.floor("h")
    old_labels = set(label for label in labels if label)
    narr_ids = sorted(old_labels, key=lambda label: (
        assigned.loc[assigned["narrativa"] == label, "ts"].min()
        if assigned.loc[assigned["narrativa"] == label, "ts"].notna().any()
        else pd.Timestamp.max.tz_localize("UTC")
    ))
    remap = {old: f"n{idx:04d}" for idx, old in enumerate(narr_ids, 1)}
    assigned["narrativa"] = assigned["narrativa"].map(remap).fillna("")
    summaries = []
    for old in narr_ids:
        new = remap[old]
        group = assigned[assigned["narrativa"] == new]
        support = group[group["postura"] == "apoya"]
        origin_pool = group[group["ts"].notna()].sort_values("ts")
        origin = pd.NaT
        if not origin_pool.empty:
            times = origin_pool["ts"].tolist()
            right = 0
            for left, candidate in enumerate(times):
                right = max(right, left)
                limit = candidate + pd.Timedelta(hours=6)
                while right < len(times) and times[right] <= limit:
                    right += 1
                if right - left >= 5:
                    origin = candidate
                    break
            if pd.isna(origin):
                origin = origin_pool["ts"].iloc[0]
        origin_row = origin_pool[origin_pool["ts"] == origin].head(1) if pd.notna(origin) else pd.DataFrame()
        origin_post = origin_row["publicacion_id"].iloc[0] if not origin_row.empty else ""
        texts = support["norm"].loc[support["norm"].str.len() > 0]
        keywords, title = [], ""
        if not texts.empty:
            try:
                vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=list(STOPWORDS), min_df=1)
                matrix = vectorizer.fit_transform(texts.tolist())
                means = np.asarray(matrix.mean(axis=0)).ravel()
                terms = vectorizer.get_feature_names_out()
                keywords = [str(terms[i]) for i in np.argsort(-means) if str(terms[i]) not in STOPWORDS][:8]
            except ValueError:
                pass
            title = min(texts.tolist(), key=lambda value: (abs(len(value) - 70), value))[:180]
        if pd.notna(origin):
            origin = origin.tz_convert("-03:00")
            origin_text = origin.isoformat()
        else:
            origin_text = ""
        summaries.append({
            "narrativa": new, "titulo": title or "Narrativa emergente",
            "frases_clave": "|".join(keywords), "fecha_origen": origin_text,
            "publicacion_origen": origin_post,
        })
    return assigned, pd.DataFrame(summaries, columns=[
        "narrativa", "titulo", "frases_clave", "fecha_origen", "publicacion_origen"
    ])


def account_features(comments: pd.DataFrame, accounts: pd.DataFrame) -> tuple[dict[str, str], pd.DataFrame]:
    if comments.empty:
        return {}, pd.DataFrame()
    account_ids = set(comments["cuenta_id"].fillna("").astype(str)) | set(accounts["cuenta_id"].fillna("").astype(str))
    account_ids.discard("")
    account_meta = accounts.drop_duplicates("cuenta_id", keep="first").set_index("cuenta_id")
    period_start = comments["ts"].dropna().min()
    global_night = comments["ts"].dt.hour.between(0, 4).mean()
    reply_edges = Counter()
    id_to_row = {}
    for idx, comment_id in enumerate(comments["comentario_id"].astype(str)):
        id_to_row.setdefault(comment_id, idx)
    account_ids_by_row = comments["cuenta_id"].fillna("").astype(str).tolist()
    for idx, row in enumerate(comments.itertuples(index=False)):
        parent_idx = id_to_row.get(str(getattr(row, "respuesta_a")))
        if parent_idx is not None:
            left, right = account_ids_by_row[idx], account_ids_by_row[parent_idx]
            if left and right and left != right:
                reply_edges[tuple(sorted((left, right)))] += 1
    pair_posts: dict[tuple[str, str], set[str]] = defaultdict(set)
    events = comments[comments["ts"].notna()].sort_values("ts")
    for post_id, post in events.groupby("publicacion_id", sort=False):
        per_account = post.groupby("cuenta_id", sort=False)["ts"].min().dropna().sort_values()
        if len(per_account) < 2:
            continue
        event_rows = [(str(account), timestamp) for account, timestamp in per_account.items()]
        buckets = sorted({timestamp.floor("5min") for _, timestamp in event_rows})
        for bucket in buckets:
            active = [account for account, timestamp in event_rows
                      if bucket <= timestamp <= bucket + timedelta(minutes=20)]
            if len(active) > 45:
                active = sorted(active)[:45]
            for pair in combinations(sorted(set(active)), 2):
                pair_posts[pair].add(str(post_id))
    text_pairs: dict[tuple[str, str], set[str]] = defaultdict(set)
    long = comments[comments["norm"].str.split().str.len().fillna(0) >= 5]
    for (norm, post_id), rows in long.groupby(["norm", "publicacion_id"], sort=False):
        users = sorted(set(rows["cuenta_id"].astype(str)))
        if len(users) < 2 or len(users) > 40:
            continue
        for pair in combinations(users, 2):
            text_pairs[pair].add(str(post_id))
    hashtag_rows = comments.loc[comments["hashtags"].fillna("").ne(""), ["hashtags", "cuenta_id", "publicacion_id"]].copy()
    hashtag_rows["tag"] = hashtag_rows["hashtags"].str.split()
    hashtag_rows = hashtag_rows.explode("tag")
    hashtag_pairs: dict[tuple[str, str], set[str]] = defaultdict(set)
    for _, rows in hashtag_rows.groupby("tag", sort=False):
        users = sorted(set(rows["cuenta_id"].astype(str)))
        if len(users) < 2 or len(users) > 100:
            continue
        post_ids = set(rows["publicacion_id"].astype(str))
        if len(post_ids) < 2:
            continue
        for pair in combinations(users, 2):
            hashtag_pairs[pair].update(post_ids)
    graph = nx.Graph()
    graph.add_nodes_from(account_ids)
    for pair, posts in pair_posts.items():
        if len(posts) >= 3:
            graph.add_edge(*pair, weight=1.0 + math.log1p(len(posts)), coactivity=len(posts))
    for pair, posts in text_pairs.items():
        if len(posts) >= 2:
            old = graph[pair[0]][pair[1]]["weight"] if graph.has_edge(*pair) else 0.0
            graph.add_edge(*pair, weight=old + 1.5 + math.log1p(len(posts)), text_reuse=len(posts))
    for pair, posts in hashtag_pairs.items():
        if len(posts) >= 2:
            old = graph[pair[0]][pair[1]]["weight"] if graph.has_edge(*pair) else 0.0
            graph.add_edge(*pair, weight=old + 1.0 + math.log1p(len(posts)), shared_hashtag=len(posts))
    for pair, count in reply_edges.items():
        if count >= 2 and graph.has_edge(*pair):
            graph[pair[0]][pair[1]]["weight"] += min(3.0, 0.5 * count)
            graph[pair[0]][pair[1]]["replies"] = count
    candidates = [component for component in nx.community.louvain_communities(graph, weight="weight", seed=SEED)
                  if 5 <= len(component) <= 250]
    coordinated, evidence_rows = {}, []
    for members in sorted(candidates, key=lambda group: min(group)):
        member_set = set(members)
        rows = comments[comments["cuenta_id"].astype(str).isin(member_set)]
        if rows.empty:
            continue
        creation_dates = []
        for account in members:
            value = account_meta.loc[account, "fecha_creacion"] if account in account_meta.index else ""
            parsed = pd.to_datetime(value, errors="coerce", utc=True)
            if pd.notna(parsed):
                creation_dates.append(parsed)
        recent_share = (
            sum(date >= period_start - pd.Timedelta(days=14) for date in creation_dates) / max(1, len(members))
            if pd.notna(period_start) else 0.0
        )
        night_share = float(rows["ts"].dt.hour.between(0, 4).mean())
        activity_times = rows["ts"].dropna().sort_values().astype("int64").to_numpy() / 1e9
        intervals = np.diff(activity_times)
        burstiness = float(np.std(intervals) / np.mean(intervals)) if len(intervals) > 1 and np.mean(intervals) > 0 else 0.0
        likes_mean = pd.to_numeric(rows.get("me_gusta", pd.Series(dtype=float)), errors="coerce").mean()
        average_likes = float(likes_mean) if pd.notna(likes_mean) else 0.0
        followers, following = [], []
        for account in members:
            if account in account_meta.index:
                followers.append(pd.to_numeric(account_meta.loc[account].get("seguidores"), errors="coerce"))
                following.append(pd.to_numeric(account_meta.loc[account].get("siguiendo"), errors="coerce"))
        median_followers = float(np.nanmedian(followers)) if followers and not np.isnan(followers).all() else 0.0
        median_following = float(np.nanmedian(following)) if following and not np.isnan(following).all() else 0.0
        text_frequency = sum(
            len(group) for _, group in rows.groupby("norm", sort=False)
            if len(set(group["cuenta_id"].astype(str))) >= 2
        )
        duplicate_share = text_frequency / max(1, len(rows))
        member_replies = sum(count for pair, count in reply_edges.items()
                             if pair[0] in member_set and pair[1] in member_set)
        reply_share = member_replies / max(1, rows["respuesta_a"].fillna("").astype(str).ne("").sum())
        density = graph.subgraph(member_set).number_of_edges() / max(1, len(member_set) * (len(member_set) - 1) / 2)
        labels = rows["narrativa"].replace("", np.nan).dropna()
        narrative_focus = float(labels.value_counts(normalize=True).iloc[0]) if len(labels) else 0.0
        score = 1.5 * recent_share + 1.25 * min(1.0, duplicate_share * 3)
        score += max(0.0, night_share - global_night * 1.5) + min(1.0, reply_share * 2)
        flagged = density >= 0.10 and score >= 1.2 and (
            recent_share >= 0.25 or duplicate_share >= 0.10
            or (reply_share >= 0.15 and night_share >= global_night) or narrative_focus >= 0.85
        )
        evidence = {
            "cuentas": len(members), "comentarios": len(rows), "coactividad_density": round(density, 4),
            "cuentas_recientes_share": round(recent_share, 4),
            "actividad_nocturna_share": round(night_share, 4),
            "reuso_texto_share": round(duplicate_share, 4),
            "respuestas_internas_share": round(reply_share, 4),
            "foco_narrativo": round(narrative_focus, 4), "interarribo_cv": round(burstiness, 4),
            "me_gusta_promedio": round(average_likes, 2),
            "seguidores_mediana": round(median_followers, 2), "siguiendo_mediana": round(median_following, 2),
            "score": round(score, 4),
            "motivo": "sincronía repetida, reuso textual, antigüedad, horario y respuestas internas",
        }
        label = f"op{len(evidence_rows) + 1:03d}" if flagged else ""
        if flagged:
            coordinated.update({str(account): label for account in members})
        evidence_rows.append({
            "grupo": label or f"candidato_{len(evidence_rows) + 1:03d}", "marcado": bool(flagged),
            "primera_actividad": rows["ts"].min().isoformat() if rows["ts"].notna().any() else "",
            "narrativa_principal": labels.value_counts().index[0] if len(labels) else "",
            "evidencia": json.dumps(evidence, ensure_ascii=False, sort_keys=True), **evidence,
        })
    return coordinated, pd.DataFrame(evidence_rows)


def hardware_description() -> str:
    cpus = os.cpu_count() or 1
    memory_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3), 1) if hasattr(os, "sysconf") else 0
    cpu = platform.processor() or platform.machine()
    return f"{cpu}; {cpus} CPU lógicos; {memory_gb} GB RAM; sin GPU"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", type=Path, default=Path("datos/practica"))
    parser.add_argument("--salida", "--entrega", dest="salida", type=Path, default=Path("entrega"))
    parser.add_argument("--embedding", choices=("auto", "minilm", "char"), default="auto")
    args = parser.parse_args()
    overall_start = time.perf_counter()
    args.salida.mkdir(parents=True, exist_ok=True)
    cache_dir = args.salida / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_root = Path(__file__).resolve().parents[1] / "modelos"
    con = duckdb.connect(str(args.salida / "nexo.duckdb"))
    stage = time.perf_counter()
    comments_input = find_input(args.datos, "comentarios")
    accounts_input = find_input(args.datos, "cuentas")
    posts_input = find_input(args.datos, "publicaciones")
    import_csv(con, comments_input, "comments_source", cache_dir / "comments_source.parquet")
    import_csv(con, accounts_input, "accounts_source", cache_dir / "accounts_source.parquet")
    import_csv(con, posts_input, "posts_source", cache_dir / "posts_source.parquet")
    comments = con.execute("SELECT * FROM comments_source").df()
    accounts = con.execute("SELECT * FROM accounts_source").df()
    con.execute("SELECT COUNT(*) FROM posts_source").fetchone()
    log_stage("ingesta DuckDB + Parquet", stage)

    stage = time.perf_counter()
    for column in ("comentario_id", "publicacion_id", "cuenta_id", "fecha", "texto", "respuesta_a", "me_gusta"):
        if column not in comments:
            comments[column] = ""
    comments["texto"] = comments["texto"].fillna("").astype(str)
    comments["respuesta_a"] = comments["respuesta_a"].fillna("").astype(str)
    normalized = [normalize_text(text) for text in comments["texto"]]
    comments["norm"] = [item[0] for item in normalized]
    comments["hashtags"] = [item[1] for item in normalized]
    prefixes, suffixes = detect_fillers(comments.loc[comments["norm"].ne(""), "norm"].tolist())
    comments["norm"] = comments["norm"].map(lambda value: strip_fillers(value, prefixes, suffixes))
    comments["ts"] = pd.to_datetime(comments["fecha"], errors="coerce", utc=True)
    comments["is_reply"] = comments["respuesta_a"].ne("")
    comments["is_noise"] = False
    nonreply_positions = np.flatnonzero(~comments["is_reply"].to_numpy())
    nonreply_noise = classify_noise(comments.iloc[nonreply_positions].reset_index(drop=True))
    comments.loc[nonreply_positions, "is_noise"] = nonreply_noise
    token_counts = comments["norm"].str.split().str.len().fillna(0)
    short_reply = comments["is_reply"] & (token_counts <= SHORT_AGREEMENT_MAX_TOKENS) & (comments["norm"].str.len() <= SHORT_AGREEMENT_MAX_CHARS)
    embedding_rows = ~comments["is_noise"] & ~short_reply
    unique_texts = sorted(set(comments.loc[embedding_rows, "norm"]) - {""})
    text_to_position = {text: idx for idx, text in enumerate(unique_texts)}
    comments["text_position"] = comments["norm"].map(text_to_position).fillna(-1).astype(int)
    log_stage("normalización, fillers y ruido", stage)

    stage = time.perf_counter()
    features, embedding_choice = make_features(unique_texts, args.embedding, model_root)
    text_labels, centers = microcluster(features)
    text_cluster = dict(zip(unique_texts, text_labels))
    cluster_rows = comments[comments["norm"].isin(text_cluster)].copy().reset_index(drop=False)
    cluster_rows["_cluster"] = cluster_rows["norm"].map(text_cluster).astype(int)
    communities = choose_narrative_communities(
        cluster_rows, cluster_rows["_cluster"].to_numpy(), centers,
        int(text_labels.max() + 1) if len(text_labels) else 0,
    )
    initial = [
        str(communities.get(int(text_cluster[text]), int(text_cluster[text]))) if text in text_cluster else ""
        for text in comments["norm"]
    ]
    labels, stances = resolve_replies(comments, initial, comments["is_noise"].to_numpy())
    unresolved_replies = [idx for idx in range(len(comments)) if comments.iloc[idx]["is_reply"] and not labels[idx]]
    if unresolved_replies:
        post_labels: dict[str, Counter[str]] = defaultdict(Counter)
        for idx, label in enumerate(labels):
            if label:
                post_labels[str(comments.iloc[idx]["publicacion_id"])][label] += 1
        global_label = Counter(label for label in labels if label).most_common(1)
        for idx in unresolved_replies:
            post_id = str(comments.iloc[idx]["publicacion_id"])
            post_label = post_labels[post_id].most_common(1)
            labels[idx] = post_label[0][0] if post_label else (global_label[0][0] if global_label else "orphan")
    micro_labels = np.asarray([int(text_cluster.get(text, 0)) for text in comments["norm"]])
    cluster_stances = assign_stances(comments, micro_labels, labels)
    for idx, label in enumerate(labels):
        if label and not comments.iloc[idx]["is_reply"]:
            stances[idx] = cluster_stances.get(micro_labels[idx], "apoya")
    stances = propagate_reply_stances(comments, labels, stances)
    assigned, narrative_table = narrative_outputs(comments, labels, stances)
    log_stage(f"embeddings/clustering ({embedding_choice})", stage)

    stage = time.perf_counter()
    if "cuenta_id" not in accounts:
        accounts["cuenta_id"] = ""
    if "fecha_creacion" not in accounts:
        accounts["fecha_creacion"] = ""
    coordinated, group_table = account_features(assigned, accounts)
    all_accounts = (set(accounts["cuenta_id"].fillna("").astype(str)) |
                    set(assigned["cuenta_id"].fillna("").astype(str))) - {""}
    account_rows = pd.DataFrame({"cuenta_id": sorted(all_accounts)})
    account_rows["grupo"] = account_rows["cuenta_id"].map(coordinated).fillna("")
    assigned["grupo"] = assigned["cuenta_id"].astype(str).map(coordinated).fillna("")
    assigned["is_coordinated"] = assigned["grupo"].ne("")
    assigned["group"] = assigned["grupo"]
    assigned["narrative"] = assigned["narrativa"].replace("", "ninguna")
    assigned["stance"] = assigned["postura"]
    assigned["narrativa"] = assigned["narrativa"].replace("", "ninguna")
    assigned["postura"] = np.where(assigned["narrativa"] == "ninguna", "", assigned["postura"])
    assigned[["comentario_id", "narrativa", "postura"]].to_csv(args.salida / "comentarios.csv", index=False)
    narrative_table[["narrativa", "titulo", "frases_clave", "fecha_origen"]].to_csv(
        args.salida / "narrativas.csv", index=False
    )
    account_rows.to_csv(args.salida / "cuentas.csv", index=False)
    log_stage("coordinación y archivos de entrega", stage)

    stage = time.perf_counter()
    assigned["hour"] = assigned["ts"].dt.floor("h")
    hourly = (assigned[assigned["hour"].notna() & assigned["narrativa"].ne("ninguna")]
              .groupby(["narrativa", "hour"], as_index=False).size()
              .rename(columns={"size": "comentarios"}))
    con.register("comments_output_df", assigned)
    con.register("narratives_output_df", narrative_table)
    con.register("groups_output_df", group_table if not group_table.empty else pd.DataFrame(
        columns=["grupo", "marcado", "primera_actividad", "narrativa_principal", "evidencia"]
    ))
    con.register("hourly_output_df", hourly)
    con.execute("CREATE OR REPLACE TABLE comments AS SELECT * FROM comments_output_df")
    con.execute("CREATE OR REPLACE TABLE narratives AS SELECT * FROM narratives_output_df")
    con.execute("CREATE OR REPLACE TABLE groups AS SELECT * FROM groups_output_df")
    con.execute("CREATE OR REPLACE TABLE hourly_counts AS SELECT * FROM hourly_output_df")
    con.execute(f"COPY comments TO {sql_path(cache_dir / 'comments_enriched.parquet')} (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.execute(f"COPY narratives TO {sql_path(cache_dir / 'narratives.parquet')} (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.execute(f"COPY groups TO {sql_path(cache_dir / 'groups.parquet')} (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.execute(f"COPY hourly_counts TO {sql_path(cache_dir / 'hourly_counts.parquet')} (FORMAT PARQUET, COMPRESSION ZSTD)")
    log_stage("tablas DuckDB y agregados horarios", stage)
    elapsed = time.perf_counter() - overall_start
    cost = {
        "comentarios_procesados": int(len(comments)), "segundos": round(elapsed, 2),
        "tokens_ia": 0, "usd_ia": 0.0, "usd_por_mil_comentarios": 0.0,
        "hardware": hardware_description(), "almacenamiento": "DuckDB local sobre Parquet",
        "notas": f"Embeddings locales ({embedding_choice}); sin llamadas a servicios ni IA generativa; semillas fijas.",
    }
    (args.salida / "costo.json").write_text(json.dumps(cost, ensure_ascii=False, indent=2), encoding="utf-8")
    con.close()
    print(f"Listo: {len(narrative_table)} narrativas, {len(set(coordinated.values()))} grupos coordinados, {elapsed:.1f}s")


if __name__ == "__main__":
    main()

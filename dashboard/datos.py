"""Carga y agregados del dashboard. Lee los datos crudos y la salida del pipeline con DuckDB."""
import json
from pathlib import Path

import duckdb
import pandas as pd


def _csv(path: Path) -> str:
    for p in (path, path.with_suffix(path.suffix + ".gz")):
        if p.exists():
            return str(p)
    gz = Path(str(path).replace(".csv.gz", ".csv"))
    if gz.exists():
        return str(gz)
    raise FileNotFoundError(path)


def cargar(datos: Path, salida: Path) -> dict:
    con = duckdb.connect()
    rd = lambda p: f"read_csv('{_csv(p)}', all_varchar=true, header=true)"
    df = con.execute(f"""
        SELECT c.comentario_id, c.publicacion_id, c.cuenta_id, COALESCE(c.texto, '') AS texto,
               COALESCE(c.respuesta_a, '') AS respuesta_a,
               TRY_CAST(c.me_gusta AS INTEGER) AS me_gusta,
               TRY_CAST(c.fecha AS TIMESTAMPTZ) AS fecha,
               p.fuente_id, f.nombre AS fuente, p.titular,
               COALESCE(NULLIF(lower(trim(o.narrativa)), 'ninguna'), '') AS narrativa,
               COALESCE(o.postura, '') AS postura,
               COALESCE(g.grupo, '') AS grupo,
               TRY_CAST(a.fecha_creacion AS DATE) AS cuenta_creada
        FROM {rd(datos/'comentarios.csv.gz')} c
        LEFT JOIN {rd(datos/'publicaciones.csv.gz')} p USING (publicacion_id)
        LEFT JOIN {rd(datos/'fuentes.csv')} f USING (fuente_id)
        LEFT JOIN {rd(salida/'comentarios.csv')} o USING (comentario_id)
        LEFT JOIN {rd(salida/'cuentas.csv')} g USING (cuenta_id)
        LEFT JOIN {rd(datos/'cuentas.csv.gz')} a USING (cuenta_id)
    """).df()
    df["fecha"] = pd.to_datetime(df["fecha"], utc=True).dt.tz_convert("America/Argentina/Buenos_Aires")
    df["me_gusta"] = df["me_gusta"].fillna(0)
    df["coord"] = df["grupo"] != ""
    narr = pd.read_csv(_csv(salida / "narrativas.csv"), dtype=str, keep_default_na=False)
    narr["narrativa"] = narr["narrativa"].str.strip().str.lower()
    narr["fecha_origen"] = pd.to_datetime(narr["fecha_origen"], utc=True, errors="coerce").dt.tz_convert(
        "America/Argentina/Buenos_Aires")
    fuentes = pd.read_csv(_csv(datos / "fuentes.csv"), dtype=str, keep_default_na=False)
    costo_path = salida / "costo.json"
    costo = json.loads(costo_path.read_text(encoding="utf-8")) if costo_path.exists() else {}
    tiempos = {}
    for name in ("tiempos.json", "etapas.json"):
        if (salida / name).exists():
            tiempos = json.loads((salida / name).read_text(encoding="utf-8"))
    tam = {p.name: p.stat().st_size for p in list(datos.glob("*")) if p.is_file()}
    pq = {p.name: p.stat().st_size for p in list(salida.rglob("*.parquet")) + list(salida.rglob("*.duckdb"))}
    return dict(df=df, narr=narr, fuentes=fuentes, costo=costo, tiempos=tiempos, tam_datos=tam, tam_pq=pq)


def resumen_narrativas(df: pd.DataFrame, narr: pd.DataFrame) -> pd.DataFrame:
    n = df[df.narrativa != ""]
    g = n.groupby("narrativa")
    t = pd.DataFrame({
        "comentarios": g.size(),
        "cuentas": g.cuenta_id.nunique(),
        "apoya": g.postura.apply(lambda s: (s == "apoya").mean()),
        "coord": g.coord.mean(),
        "primer": g.fecha.min(),
        "fuente_top": g.fuente.agg(lambda s: s.value_counts().index[0] if len(s.dropna()) else ""),
    })
    cn = n[n.coord].groupby(["grupo", "narrativa"]).size().rename("k").reset_index()
    cn["share"] = cn.k / cn.groupby("grupo").k.transform("sum")
    t["grupos"] = cn[cn.share >= 0.2].groupby("narrativa").grupo.agg(lambda s: ", ".join(sorted(s)))
    t = t.join(narr.set_index("narrativa")[["titulo", "frases_clave", "fecha_origen"]], how="left")
    t["titulo"] = t["titulo"].fillna(t.index.to_series())
    t["fecha_origen"] = t["fecha_origen"].fillna(t["primer"])
    t["grupos"] = t["grupos"].fillna("")
    return t.sort_values("comentarios", ascending=False)


def ojo_organico(df: pd.DataFrame, narrativa: str, primeros: int = 20) -> dict:
    n = df[df.narrativa == narrativa].sort_values("fecha")
    if n.empty:
        return {}
    first = n.head(primeros)
    first_day = n[n.fecha < n.fecha.iloc[0] + pd.Timedelta(hours=24)]
    org = n[~n.coord]
    return dict(
        total=len(n), coord=int(n.coord.sum()), share=n.coord.mean(),
        primeros_coord=int(first.coord.sum()), primeros=len(first),
        dia1_share=first_day.coord.mean(),
        likes_coord=n[n.coord].me_gusta.mean() if n.coord.any() else 0.0,
        likes_org=org.me_gusta.mean() if len(org) else 0.0,
        likes_share=n[n.coord].me_gusta.sum() / max(1, n.me_gusta.sum()),
        primero=n.iloc[0], primero_org=org.iloc[0] if len(org) else None,
        grupos=n[n.coord].grupo.value_counts(),
    )


def perfil_grupos(df: pd.DataFrame) -> pd.DataFrame:
    if not df.coord.any():
        return pd.DataFrame()
    start = df.fecha.min()
    c = df[df.coord].copy()
    c["hora"] = c.fecha.dt.hour
    base_night = df.fecha.dt.hour.between(1, 5).mean()
    rows = []
    for gname, g in c.groupby("grupo"):
        accs = g.cuenta_id.unique()
        created = pd.to_datetime(df[df.cuenta_id.isin(accs)].drop_duplicates("cuenta_id").cuenta_creada)
        age = (start.tz_localize(None).normalize() - created).dt.days
        narr = g[g.narrativa != ""].narrativa.value_counts()
        ids = set(g.comentario_id)
        intra = g.respuesta_a.isin(ids).mean()
        rows.append(dict(
            grupo=gname, cuentas=len(accs), comentarios=len(g),
            narrativa=narr.index[0] if len(narr) else "", foco=(narr.iloc[0] / narr.sum()) if len(narr) else 0,
            postura=g.postura.value_counts().index[0] if (g.postura != "").any() else "",
            antiguedad_mediana_dias=float(age.median()) if len(age.dropna()) else None,
            nocturno=g.hora.between(1, 5).mean(), nocturno_base=base_night,
            textos_unicos=g.texto.nunique() / len(g), respuestas_internas=intra,
            likes=g.me_gusta.mean(), publicaciones=g.publicacion_id.nunique(),
            primera=g.fecha.min(), ultima=g.fecha.max()))
    return pd.DataFrame(rows).sort_values("cuentas", ascending=False)


def alertas(df: pd.DataFrame, rn: pd.DataFrame, grupos: pd.DataFrame) -> pd.DataFrame:
    a = []
    titulos = rn["titulo"].to_dict()
    start = df.fecha.min()
    for nid, r in rn.iterrows():
        if pd.notna(r.fecha_origen) and r.fecha_origen > start + pd.Timedelta(hours=36):
            a.append(dict(fecha=r.fecha_origen, tipo="Narrativa emergente", sev="Alta",
                          texto=f"Nueva narrativa «{r.titulo}»: {r.comentarios:,} comentarios desde su origen."))
        lens = ojo_organico(df, nid)
        if lens and lens["primeros_coord"] >= lens["primeros"] / 2:
            a.append(dict(fecha=lens["primero"].fecha, tipo="Narrativa sembrada", sev="Crítica",
                          texto=f"«{r.titulo}»: {lens['primeros_coord']} de los primeros {lens['primeros']} "
                                f"comentarios vienen de cuentas coordinadas."))
    for _, g in grupos.iterrows():
        a.append(dict(fecha=g.primera, tipo="Operación coordinada", sev="Crítica" if g.cuentas >= 20 else "Alta",
                      texto=f"Grupo {g.grupo}: {g.cuentas} cuentas, {g.comentarios} comentarios en "
                            f"{g.publicaciones} publicaciones; empuja «{titulos.get(g.narrativa, g.narrativa) or 'varias'}»."))
    h = df.set_index("fecha").groupby(pd.Grouper(freq="1h")).agg(n=("comentario_id", "size"),
                                                                 cuentas=("cuenta_id", "nunique"),
                                                                 coord=("coord", "mean"))
    if len(h):
        thr = 4 * max(1.0, h.n[h.n > 0].median())
        for t, r in h[h.n > thr].sort_values("n", ascending=False).head(3).iterrows():
            genuino = r.coord < 0.02 and r.cuentas > 0.6 * r.n
            a.append(dict(fecha=t, tipo="Pico viral" + (" genuino" if genuino else ""),
                          sev="Info" if genuino else "Media",
                          texto=f"{int(r.n):,} comentarios en una hora de {int(r.cuentas):,} cuentas distintas "
                                f"({r.coord:.1%} coordinadas)."))
    out = pd.DataFrame(a)
    return out.sort_values("fecha", ascending=False) if len(out) else out

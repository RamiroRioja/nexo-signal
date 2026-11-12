"""NEXO SIGNAL · dashboard. Uso:
    streamlit run dashboard/app.py -- --datos datos/practica --salida salida/practica
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
import datos as D  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--datos", default="datos/final")
ap.add_argument("--salida", default="entrega")
args, _ = ap.parse_known_args()

st.set_page_config(page_title="NEXO SIGNAL", page_icon="◉", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.6rem}
div[data-testid="stMetric"]{background:#121a3a;border:1px solid #26306a;border-radius:12px;padding:10px 14px}
.card{background:#121a3a;border:1px solid #26306a;border-radius:12px;padding:12px 14px;margin-bottom:8px}
div[data-testid="stMetricValue"]{font-size:1.55rem}
.q{border-left:3px solid #818cf8;padding:6px 10px;margin:4px 0;background:#0d1430;border-radius:6px;font-size:.92rem}
.q.r{border-color:#f43f5e}.q.c{border-color:#f59e0b}
.tag{display:inline-block;padding:1px 8px;border-radius:99px;font-size:.75rem;margin-right:4px;background:#26306a}
.sev-Crítica{background:#7f1d1d}.sev-Alta{background:#7c2d12}.sev-Media{background:#713f12}.sev-Info{background:#1e3a8a}
</style>""", unsafe_allow_html=True)
PLOT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=30, b=10))


@st.cache_data(show_spinner="Cargando datos con DuckDB…")
def load(datos, salida):
    d = D.cargar(Path(datos), Path(salida))
    d["rn"] = D.resumen_narrativas(d["df"], d["narr"])
    d["grupos"] = D.perfil_grupos(d["df"])
    d["alertas"] = D.alertas(d["df"], d["rn"], d["grupos"])
    return d


try:
    d = load(args.datos, args.salida)
except FileNotFoundError as e:
    st.error(f"No encuentro {e}. Corré primero el pipeline: `python -m nexo --datos {args.datos} --salida {args.salida}`")
    st.stop()
df, rn, grupos, al = d["df"], d["rn"], d["grupos"], d["alertas"]
label = lambda n: f"{rn.loc[n, 'titulo']}" if n in rn.index else n

with st.sidebar:
    st.markdown("## ◉ NEXO SIGNAL\n<small>radar de conversaciones</small>", unsafe_allow_html=True)
    tab = st.radio("Sección", ["Fuentes", "Panorama", "Narrativas", "Organic Lens", "Coordinación", "Alertas",
                               "Costos"], label_visibility="collapsed")
    st.caption(f"Datos: `{args.datos}`  \nSalida: `{args.salida}`  \n"
               f"Período: {df.fecha.min():%d/%m} – {df.fecha.max():%d/%m}")
    st.caption("Datos sintéticos del desafío. Las conexiones a redes son simuladas.")

# ---------------------------------------------------------------- Fuentes
if tab == "Fuentes":
    st.title("Fuentes conectadas")
    st.caption("Hoy: conector CSV del desafío. Roadmap: APIs oficiales de Meta, X, TikTok y YouTube "
               "sobre páginas propias del cliente y con su consentimiento.")
    f = df.groupby(["fuente_id", "fuente"]).agg(publicaciones=("publicacion_id", "nunique"),
                                                comentarios=("comentario_id", "size"),
                                                cuentas=("cuenta_id", "nunique"),
                                                coordinadas=("coord", "mean")).reset_index()
    f = f.merge(d["fuentes"][["fuente_id", "tipo"]], on="fuente_id", how="left")
    cols = st.columns(4)
    for i, r in f.sort_values("comentarios", ascending=False).reset_index(drop=True).iterrows():
        with cols[i % 4]:
            st.markdown(f"""<div class="card"><b>{r.fuente}</b><br><span class="tag">{r.tipo}</span>
            <span class="tag" style="background:#14532d">conectada · CSV</span><br>
            <small>{r.publicaciones:,} publicaciones · {r.comentarios:,} comentarios · {r.cuentas:,} cuentas ·
            {r.coordinadas:.1%} coordinado</small></div>""", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Comentarios", f"{len(df):,}")
    c2.metric("Publicaciones", f"{df.publicacion_id.nunique():,}")
    c3.metric("Cuentas", f"{df.cuenta_id.nunique():,}")
    c4.metric("Tiempo de análisis", f"{d['costo'].get('segundos', 0) / 60:.1f} min")

# ---------------------------------------------------------------- Panorama
elif tab == "Panorama":
    st.title("Panorama")
    st.caption("Qué circula, desde cuándo y quién lo empuja.")
    c = st.columns(5)
    c[0].metric("Comentarios", f"{len(df):,}")
    c[1].metric("Narrativas", len(rn))
    c[2].metric("Ruido", f"{(df.narrativa == '').mean():.0%}")
    c[3].metric("Operaciones coordinadas", len(grupos))
    c[4].metric("Cuentas coordinadas", f"{int(grupos.cuentas.sum()) if len(grupos) else 0:,}")
    n = df[df.narrativa != ""].copy()
    n["dia"] = n.fecha.dt.floor("D")
    s = n.groupby(["dia", "narrativa"]).size().reset_index(name="comentarios")
    s["titulo"] = s.narrativa.map(label)
    fig = px.area(s, x="dia", y="comentarios", color="titulo")
    for nid, r in rn.iterrows():
        if pd.notna(r.fecha_origen) and r.fecha_origen > df.fecha.min() + pd.Timedelta(hours=36):
            fig.add_vline(x=r.fecha_origen, line_dash="dot", line_color="#f59e0b")
            fig.add_annotation(x=r.fecha_origen, y=1.04, yref="paper", text="nace", showarrow=False,
                               font=dict(size=10, color="#f59e0b"), hovertext=r.titulo)
    fig.update_layout(height=380, legend=dict(font=dict(size=10)), **PLOT)
    st.plotly_chart(fig, use_container_width=True)
    t = rn.reset_index()[["titulo", "comentarios", "cuentas", "apoya", "fecha_origen", "fuente_top", "grupos", "coord"]]
    t["fecha_origen"] = t.fecha_origen.dt.strftime("%d/%m %H:%M")
    st.dataframe(t, hide_index=True, use_container_width=True, column_config={
        "titulo": "Narrativa", "apoya": st.column_config.ProgressColumn("Apoya", format="percent", min_value=0, max_value=1),
        "coord": st.column_config.NumberColumn("% coordinado", format="percent"),
        "fecha_origen": "Origen", "fuente_top": "Fuente principal", "grupos": "Empujada por"})

# ---------------------------------------------------------------- Narrativas
elif tab == "Narrativas":
    st.title("Narrativas")
    nid = st.selectbox("Narrativa", rn.index, format_func=label)
    r = rn.loc[nid]
    n = df[df.narrativa == nid].sort_values("fecha")
    st.markdown(f"### «{r.titulo}»")
    st.markdown(" ".join(f'<span class="tag">{k}</span>' for k in str(r.frases_clave).split("|") if k),
                unsafe_allow_html=True)
    o = n.iloc[0]
    c = st.columns(5)
    c[0].metric("Comentarios", f"{len(n):,}")
    c[1].metric("Cuentas", f"{n.cuenta_id.nunique():,}")
    c[2].metric("Apoya / rechaza", f"{(n.postura == 'apoya').mean():.0%}/{(n.postura == 'rechaza').mean():.0%}")
    c[3].metric("Son respuestas", f"{(n.respuesta_a != '').mean():.0%}")
    c[4].metric("Origen", f"{r.fecha_origen:%d/%m %H:%M}" if pd.notna(r.fecha_origen) else "–")
    st.caption(f"Primer comentario en **{o.fuente}** · publicación «{o.titular}»")
    left, right = st.columns([3, 2])
    with left:
        s = n.groupby([n.fecha.dt.floor("D"), "postura"]).size().reset_index(name="comentarios")
        fig = px.bar(s, x="fecha", y="comentarios", color="postura",
                     color_discrete_map={"apoya": "#22c55e", "rechaza": "#f43f5e"}, title="Postura por día")
        fig.update_layout(height=300, **PLOT)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fs = n.fuente.value_counts().reset_index()
        fig = px.bar(fs, x="count", y="fuente", orientation="h", title="Dónde circula")
        fig.update_layout(height=300, yaxis=dict(autorange="reversed"), **PLOT)
        st.plotly_chart(fig, use_container_width=True)
    own = n[(n.respuesta_a == "") & (n.texto.str.len() > 30)]
    mid = n.fecha.iloc[0] + (n.fecha.iloc[-1] - n.fecha.iloc[0]) / 2
    a, b, cc = st.columns(3)
    with a:
        st.markdown("**Cómo se decía al principio**")
        for t in own[(own.postura == "apoya") & (own.fecha < mid)].texto.head(4):
            st.markdown(f'<div class="q">{t}</div>', unsafe_allow_html=True)
    with b:
        st.markdown("**Cómo se dice después**")
        for t in own[(own.postura == "apoya") & (own.fecha >= mid)].texto.tail(4):
            st.markdown(f'<div class="q c">{t}</div>', unsafe_allow_html=True)
    with cc:
        st.markdown("**Cómo la rechazan**")
        for t in own[own.postura == "rechaza"].texto.drop_duplicates().head(4):
            st.markdown(f'<div class="q r">{t}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- Organic Lens
elif tab == "Organic Lens":
    st.title("Organic Lens")
    st.caption("¿La conversación nació sola o fue sembrada? No dice si una afirmación es falsa: "
               "muestra señales de amplificación coordinada.")
    order = rn.sort_values("coord", ascending=False).index
    nid = st.selectbox("Narrativa", order, format_func=lambda x: f"{label(x)}  ·  {rn.loc[x, 'coord']:.1%} coordinado")
    L = D.ojo_organico(df, nid)
    c = st.columns(4)
    c[0].metric(f"Coordinados en los primeros {L['primeros']}", f"{L['primeros_coord']} de {L['primeros']}")
    c[1].metric("Coordinado en el primer día", f"{L['dia1_share']:.0%}")
    c[2].metric("Me gusta: coordinado vs orgánico", f"{L['likes_coord']:.1f} vs {L['likes_org']:.1f}")
    c[3].metric("Coordinado en el volumen total", f"{L['share']:.1%}")
    n = df[df.narrativa == nid].sort_values("fecha")
    t0 = n.fecha.iloc[0]
    horas = st.slider("Ventana desde el origen (horas)", 12, 24 * 14, 72, 12)
    w = n[n.fecha < t0 + pd.Timedelta(hours=horas)]
    freq = "1h" if horas <= 96 else "6h"
    s = w.groupby([pd.Grouper(key="fecha", freq=freq), "coord"]).size().unstack(fill_value=0)
    s = s.reindex(columns=[False, True], fill_value=0)
    fig = go.Figure()
    fig.add_bar(x=s.index, y=s[False], name="orgánico", marker_color="#60a5fa")
    fig.add_bar(x=s.index, y=s[True], name="coordinado", marker_color="#f43f5e")
    fig.update_layout(barmode="stack", height=340, title="Comentarios por franja: orgánico vs coordinado", **PLOT)
    st.plotly_chart(fig, use_container_width=True)
    p = L["primero"]
    po = L["primero_org"]
    a, b = st.columns(2)
    with a:
        st.markdown(f"""<div class="card"><b>Primer comentario</b> · {p.fecha:%d/%m %H:%M} · {p.fuente}
        {'<span class="tag sev-Crítica">coordinado · ' + p.grupo + '</span>' if p.coord else '<span class="tag">orgánico</span>'}
        <div class="q">{p.texto}</div></div>""", unsafe_allow_html=True)
    with b:
        if po is not None:
            lag = (po.fecha - p.fecha).total_seconds() / 3600
            st.markdown(f"""<div class="card"><b>Primer comentario orgánico</b> · {po.fecha:%d/%m %H:%M}
            (+{lag:.1f} h) · {po.fuente}<div class="q">{po.texto}</div></div>""", unsafe_allow_html=True)
    if len(L["grupos"]):
        st.markdown("**Grupos que la empujan:** " + ", ".join(f"{g} ({v} comentarios)" for g, v in L["grupos"].items()))

# ---------------------------------------------------------------- Coordinación
elif tab == "Coordinación":
    st.title("Coordinación")
    st.caption("Cuentas que comentan juntas, en las mismas publicaciones, una y otra vez. Gris: las cuentas "
               "genuinas más activas (también coinciden, pero sin repetirse en grupo).")
    if grupos.empty:
        st.info("El pipeline no marcó operaciones coordinadas en este set.")
        st.stop()

    @st.cache_data
    def red(_df, key, ventana_min=30, min_veces=2, extra=250):
        flagged = set(_df[_df.coord].cuenta_id)
        top = _df[~_df.coord].cuenta_id.value_counts().head(extra).index
        sub = _df[_df.cuenta_id.isin(flagged | set(top))].sort_values("fecha")
        w = sub.assign(slot=sub.fecha.dt.floor(f"{ventana_min}min"))
        pairs = {}
        for _, g in w.groupby(["publicacion_id", "slot"]):
            accs = sorted(set(g.cuenta_id))
            if 2 <= len(accs) <= 60:
                for a, b in combinations(accs, 2):
                    pairs[(a, b)] = pairs.get((a, b), 0) + 1
        G = nx.Graph()
        G.add_nodes_from(flagged)
        G.add_edges_from((a, b, {"w": v}) for (a, b), v in pairs.items()
                         if v >= min_veces or (a not in flagged and b not in flagged))
        gmap = _df.drop_duplicates("cuenta_id").set_index("cuenta_id").grupo
        names = sorted({gmap.get(n, "") for n in G.nodes} - {""})
        rng = np.random.default_rng(7)
        pos = {}
        for i, gname in enumerate(names):
            ang = 2 * np.pi * i / max(1, len(names))
            cx, cy = (0.9 * np.cos(ang), 0.9 * np.sin(ang)) if len(names) > 1 else (0, 0)
            nodes = [n for n in G.nodes if gmap.get(n, "") == gname]
            sub_pos = nx.spring_layout(G.subgraph(nodes), seed=7)
            arr = np.array(list(sub_pos.values())) if sub_pos else np.zeros((1, 2))
            scale = 0.28 / max(1e-9, np.quantile(np.abs(arr), 0.9))
            for n, (x, y) in sub_pos.items():
                pos[n] = (cx + float(np.clip(x * scale, -0.4, 0.4)), cy + float(np.clip(y * scale, -0.4, 0.4)))
        for n in G.nodes:
            if n not in pos:
                r, a = rng.uniform(1.6, 2.3), rng.uniform(0, 2 * np.pi)
                pos[n] = (r * np.cos(a), r * np.sin(a))
        return G, pos

    G, pos = red(df, args.salida)
    gmap = df.drop_duplicates("cuenta_id").set_index("cuenta_id").grupo
    ex, ey = [], []
    for a, b in G.edges():
        if gmap.get(a, "") == "" or gmap.get(b, "") == "":
            continue
        ex += [pos[a][0], pos[b][0], None]
        ey += [pos[a][1], pos[b][1], None]
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=0.4, color="#334155"), hoverinfo="skip",
                               showlegend=False))
    palette = px.colors.qualitative.Bold
    for i, gname in enumerate([""] + list(grupos.grupo)):
        nodes = [n for n in G.nodes if gmap.get(n, "") == gname]
        fig.add_scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers",
                        name=gname or "genuinas (las más activas)", text=nodes, hoverinfo="text",
                        marker=dict(size=6 if not gname else 9, opacity=0.6 if not gname else 1,
                                    color="#64748b" if not gname else palette[(i - 1) % len(palette)]))
    fig.update_layout(height=560, xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"), **PLOT)
    st.plotly_chart(fig, use_container_width=True)
    g = grupos.copy()
    g["narrativa"] = g.narrativa.map(label)
    st.dataframe(g[["grupo", "cuentas", "comentarios", "publicaciones", "narrativa", "postura", "foco",
                    "antiguedad_mediana_dias", "nocturno", "textos_unicos", "respuestas_internas", "likes", "primera"]],
                 hide_index=True, use_container_width=True, column_config={
                     "foco": st.column_config.NumberColumn("Foco en 1 narrativa", format="percent"),
                     "nocturno": st.column_config.NumberColumn("Actividad 01–05 h", format="percent"),
                     "textos_unicos": st.column_config.NumberColumn("Textos únicos", format="percent"),
                     "respuestas_internas": st.column_config.NumberColumn("Se responden entre sí", format="percent"),
                     "antiguedad_mediana_dias": st.column_config.NumberColumn("Antigüedad mediana (días)",
                                                                              format="%d"),
                     "likes": st.column_config.NumberColumn("Me gusta / com.", format="%.1f")})
    st.caption(f"Base: {grupos.nocturno_base.iloc[0]:.1%} de toda la actividad ocurre entre 01 y 05 h.")

# ---------------------------------------------------------------- Alertas
elif tab == "Alertas":
    st.title("Alertas")
    if al.empty:
        st.info("Sin alertas.")
        st.stop()
    tipos = st.multiselect("Tipo", sorted(al.tipo.unique()), default=sorted(al.tipo.unique()))
    left, right = st.columns([3, 1])
    with left:
        for _, a in al[al.tipo.isin(tipos)].iterrows():
            st.markdown(f"""<div class="card"><span class="tag sev-{a.sev}">{a.sev}</span>
            <span class="tag">{a.tipo}</span> <small>{a.fecha:%d/%m %H:%M}</small><br>{a.texto}</div>""",
                        unsafe_allow_html=True)
    with right:
        st.markdown("**Reglas**")
        for rname in ["Narrativa emergente", "Narrativa sembrada", "Operación coordinada", "Pico viral",
                      "Cambio de redacción"]:
            st.toggle(rname, value=rname != "Cambio de redacción")
        st.markdown("**Canales** (roadmap)")
        for ch in ["Email", "Slack", "WhatsApp", "Webhook"]:
            st.checkbox(ch, value=ch in ("Email", "Webhook"))

# ---------------------------------------------------------------- Costos
elif tab == "Costos":
    st.title("Costos y escala")
    k = d["costo"]
    c = st.columns(5)
    c[0].metric("US$ / 1.000 com.", f"{k.get('usd_por_mil_comentarios', 0):.4f}")
    c[1].metric("Tiempo total", f"{k.get('segundos', 0) / 60:.1f} min")
    c[2].metric("Velocidad", f"{k.get('comentarios_procesados', 0) / max(1, k.get('segundos', 1)):,.0f}/s")
    c[3].metric("Tokens de IA", f"{k.get('tokens_ia', 0):,}")
    pq = sum(d["tam_pq"].values())
    c[4].metric("Almacenamiento", f"{pq / 1e6:.1f} MB" if pq else "–", help="Parquet/DuckDB en la salida")
    left, right = st.columns(2)
    with left:
        if d["tiempos"]:
            t = pd.Series(d["tiempos"], dtype=float).rename("segundos").reset_index()
            fig = px.bar(t, x="segundos", y="index", orientation="h", title="Tiempo por etapa")
            fig.update_layout(height=320, yaxis=dict(autorange="reversed", title=""), **PLOT)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("**costo.json**")
        st.json(k)
    with right:
        st.markdown("""**Por qué DuckDB + Parquet**
- Sin servidor: corre embebido en el pipeline; el jurado lo reproduce con un comando.
- Columnar: agregaciones por hora, cuenta y publicación en milisegundos.
- Parquet es abierto: el mismo archivo lo leen Spark, ClickHouse o pandas.
- Costo de infraestructura: 0.

**Cómo escala a millones de comentarios por día**
1. Conectores (APIs oficiales / CSV) → cola (Redpanda/Kafka).
2. Embeddings en workers CPU, con caché por texto deduplicado.
3. Cada comentario nuevo se asigna a la narrativa más cercana (índice vectorial).
4. Lo que no encaja se re-agrupa cada hora → narrativas nuevas.
5. Grafo de coordinación por ventana deslizante.
6. Parquet particionado por día + ClickHouse para el dashboard.

Sin IA generativa por comentario: el costo marginal es CPU.""")

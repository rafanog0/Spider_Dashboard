# app2.py
# -*- coding: utf-8 -*-
"""
VR Fobia a Aranhas — Dashboard analítico (Streamlit)
- IDATE (tabela longa) é pivotado corretamente para colunas IDATE_*
- GroupBy usa observed=True (evita categorias vazias) + remoção de categorias não usadas
- MixedLM mais estável (DV padronizado, lbfgs, reml=False)
- Tooltips didáticos com st.expander("❓ ...")
- Aba extra "Hipóteses de Pesquisa" (documentação H1–H4)
"""

import os, re, json, math, sqlite3
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import norm, mannwhitneyu, shapiro, ttest_ind

import plotly.express as px
import plotly.graph_objects as go

import statsmodels.api as sm
import statsmodels.formula.api as smf
import pingouin as pg
import streamlit as st

# ---------------------------------------
# Streamlit config
# ---------------------------------------
st.set_page_config(page_title="VR Fobia (Análise)", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------
# Utils gerais
# ---------------------------------------

def parse_session_bpm(x: Any) -> List[float]:
    """Aceita CSV/JSON/list/list[dict{bpm}]. Retorna lista de floats."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    try:
        if isinstance(x, list):
            if not x:
                return []
            if isinstance(x[0], dict):
                return [float(d.get("bpm", np.nan)) for d in x if "bpm" in d]
            return [float(v) for v in x]
        s = str(x).strip()
        if s.startswith("[") or s.startswith("{"):
            obj = json.loads(s)
            if isinstance(obj, list):
                if not obj:
                    return []
                if isinstance(obj[0], dict):
                    return [float(d.get("bpm", np.nan)) for d in obj if "bpm" in d]
                return [float(v) for v in obj]
            if isinstance(obj, dict) and "values" in obj:
                return [float(v) for v in obj["values"]]
        # CSV
        parts = re.split(r"[,\s;]+", s)
        out = []
        for p in parts:
            p = p.strip()
            try:
                out.append(float(p))
            except Exception:
                pass
        return out
    except Exception:
        return []

def bpm_metrics(series: List[float], peak_quantile: float = 0.95) -> Dict[str, float]:
    arr = np.array([v for v in series if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"bpm_mean": np.nan, "bpm_max": np.nan, "bpm_peaks": np.nan}
    mean_ = float(np.mean(arr))
    max_ = float(np.max(arr))
    thr = float(np.quantile(arr, peak_quantile))
    idx, _ = find_peaks(arr, height=thr)
    return {"bpm_mean": mean_, "bpm_max": max_, "bpm_peaks": float(len(idx))}

def fisher_r_to_z(r: float) -> float:
    return 0.5 * np.log((1 + r) / (1 - r))

def compare_correlations_fisher(r1: float, n1: int, r2: float, n2: int) -> Dict[str, float]:
    z1, z2 = fisher_r_to_z(r1), fisher_r_to_z(r2)
    se = math.sqrt(1/(n1-3) + 1/(n2-3))
    z = (z1 - z2) / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return {"z": float(z), "p": float(p)}

# ---------------------------------------
# ETL SQLite
# ---------------------------------------

@st.cache_data(show_spinner=False)
def load_sqlite_tables(db_path: str) -> Dict[str, pd.DataFrame]:
    con = sqlite3.connect(db_path)
    try:
        names = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", con)["name"].tolist()
        out = {}
        for t in names:
            try:
                out[t] = pd.read_sql(f"SELECT * FROM [{t}]", con)
            except Exception:
                pass
        return out
    finally:
        con.close()

def unify_fases(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    fase_tables = [name for name in tables.keys() if re.match(r"^fase\d+$", name)]
    frames = []
    for name in sorted(fase_tables, key=lambda s: int(re.findall(r"\d+", s)[0])):
        df = tables[name].copy()
        df["fase"] = int(re.findall(r"\d+", name)[0])
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["participante_id", "fase", "session_bpm"])
    return pd.concat(frames, ignore_index=True)

def summarize_bpm(fases_long: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    df = fases_long.copy()
    df["bpm_series"] = df["session_bpm"].apply(parse_session_bpm)
    m = df["bpm_series"].apply(bpm_metrics).apply(pd.Series)
    df = pd.concat([df, m], axis=1)
    df["bpm_metric"] = df[metric_key]
    out = (df.groupby(["participante_id", "fase"], as_index=False)
             .agg(bpm_metric=("bpm_metric", "mean"),
                  bpm_mean=("bpm_mean", "mean"),
                  bpm_max=("bpm_max", "mean"),
                  bpm_peaks=("bpm_peaks", "mean")))
    return out

def compute_idate_scores(idate: pd.DataFrame) -> pd.DataFrame:
    """
    Converte tabela longa:
      colunas: participante_id, tipo, questao, resposta
      tipo ∈ {'idate_traco','idate_estado_pre','idate_estado_pos'} (strings)
    -> pivot em colunas: 'IDATE_traco', 'IDATE_estado_pre', 'IDATE_estado_pos'
    """
    if idate is None or idate.empty:
        return pd.DataFrame(columns=["participante_id", "IDATE_traco", "IDATE_estado_pre", "IDATE_estado_pos"])
    df = idate.copy()
    df["tipo"] = df["tipo"].astype(str).str.strip().str.lower()
    mapa = {
        "idate_traco": "IDATE_traco",
        "idate_traço": "IDATE_traco",
        "traco": "IDATE_traco",
        "traço": "IDATE_traco",
        "idate_estado_pre": "IDATE_estado_pre",
        "estado_pre": "IDATE_estado_pre",
        "idate_estado_pos": "IDATE_estado_pos",
        "estado_pos": "IDATE_estado_pos",
    }
    df["tipo_std"] = df["tipo"].map(mapa)
    df = df.dropna(subset=["tipo_std", "participante_id", "resposta"])
    g = (df.groupby(["participante_id", "tipo_std"], as_index=False)
           .agg(score=("resposta", "sum")))
    pivot = g.pivot(index="participante_id", columns="tipo_std", values="score").reset_index()
    for c in ["IDATE_traco", "IDATE_estado_pre", "IDATE_estado_pos"]:
        if c not in pivot.columns:
            pivot[c] = np.nan
    # garante tipo numérico
    for c in ["IDATE_traco", "IDATE_estado_pre", "IDATE_estado_pos"]:
        pivot[c] = pd.to_numeric(pivot[c], errors="coerce")
    return pivot

def aggregate_pos_vr(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["ansiedade", "aversao_aranha", "realismo", "enjoo"] if c in df.columns]
    if not cols:
        return pd.DataFrame(columns=["participante_id"])
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    agg = (df.groupby("participante_id", as_index=False)
             .agg(**{c: (c, "mean") for c in cols}))
    return agg

def master_table(tables: Dict[str, pd.DataFrame],
                 bpm_metric_key: str,
                 include_cardiopaths: bool,
                 selected_groups: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    participantes = tables.get("participantes", pd.DataFrame())
    fases_long = unify_fases(tables)
    idate = tables.get("idate", pd.DataFrame())
    pos_vr = tables.get("pos_vr_questionario", pd.DataFrame())

    bpm = summarize_bpm(fases_long, bpm_metric_key)
    idate_scores = compute_idate_scores(idate)
    pos_vr_agg = aggregate_pos_vr(pos_vr) if not pos_vr.empty else pd.DataFrame()

    df = bpm.merge(participantes, left_on="participante_id", right_on="id", how="left")
    if not idate_scores.empty:
        df = df.merge(idate_scores, on="participante_id", how="left")
    if not pos_vr_agg.empty:
        df = df.merge(pos_vr_agg, on="participante_id", how="left")

    # tipos
    if "grupo" in df.columns:
        df["grupo"] = df["grupo"].astype("category").cat.remove_unused_categories()
    if "fase" in df.columns:
        df["fase"] = pd.to_numeric(df["fase"], errors="coerce").astype("Int64")

    # filtros
    if selected_groups and "grupo" in df.columns:
        df = df[df["grupo"].isin(selected_groups)]
    if not include_cardiopaths and "problema_cardiaco" in df.columns:
        df = df[(df["problema_cardiaco"].isna()) | (df["problema_cardiaco"] == 0)]

    # força numérico nas métricas-chave (evita surpresas)
    for col in ["bpm_mean", "bpm_max", "bpm_peaks",
                "IDATE_traco", "IDATE_estado_pre", "IDATE_estado_pos",
                "ansiedade", "aversao_aranha", "realismo", "enjoo"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, participantes

# ---------------------------------------
# Modelagem
# ---------------------------------------

def fit_lme_h1(df: pd.DataFrame, dv: str = "bpm_metric") -> Optional[sm.regression.mixed_linear_model.MixedLMResults]:
    d = df.dropna(subset=[dv, "grupo", "fase", "participante_id"]).copy()
    if d.empty:
        return None
    d["fase_c"] = d["fase"].astype("category")

    # padroniza DV para melhorar condicionamento
    mu, sd = d[dv].mean(), d[dv].std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    d["_dv_z"] = (d[dv] - mu) / sd

    try:
        model = smf.mixedlm("_dv_z ~ C(grupo)*C(fase_c)", data=d, groups=d["participante_id"])
        res = model.fit(method="lbfgs", reml=False, maxiter=1000, disp=False)
        return res
    except Exception:
        try:
            model = smf.mixedlm("_dv_z ~ C(grupo)+C(fase_c)", data=d, groups=d["participante_id"])
            return model.fit(method="lbfgs", reml=False, maxiter=1000, disp=False)
        except Exception:
            return None

def anova_mista_h1(df: pd.DataFrame, dv: str = "bpm_metric") -> Optional[pd.DataFrame]:
    d = df.dropna(subset=[dv, "grupo", "fase", "participante_id"]).copy()
    if d.empty:
        return None
    d["fase"] = d["fase"].astype("category")
    try:
        aov = pg.mixed_anova(dv=dv, within="fase", between="grupo", subject="participante_id", data=d)
        return aov
    except Exception:
        return None

def ancova_h3(df_participant: pd.DataFrame, use_covariates: List[str]) -> Optional[pd.DataFrame]:
    """
    ANCOVA robusta: 1 linha por participante -> média de PRE/POS; observed=True evita
    categorias vazias. Corrige ValueError de length mismatch.
    """
    req = ["grupo", "IDATE_estado_pre", "IDATE_estado_pos", "participante_id"]
    if not set(req).issubset(df_participant.columns):
        return None

    # Base mínima e limpa
    d0 = df_participant[req + [c for c in use_covariates if c in df_participant.columns]].copy()
    # saneia categorias
    if "grupo" in d0.columns:
        d0["grupo"] = d0["grupo"].astype("category").cat.remove_unused_categories()
    # numérico
    for c in ["IDATE_estado_pre", "IDATE_estado_pos"] + [cv for cv in use_covariates if cv in d0.columns]:
        d0[c] = pd.to_numeric(d0[c], errors="coerce")

    # 1 linha por participante: média (se houver duplicidade por fase)
    d = (d0.dropna(subset=["participante_id", "grupo"])
            .groupby(["participante_id", "grupo"], observed=True, as_index=False)
            .agg(IDATE_estado_pre=("IDATE_estado_pre", "mean"),
                 IDATE_estado_pos=("IDATE_estado_pos", "mean")))
    # agrega covariáveis (médias por participante)
    for cov in use_covariates:
        if cov in d0.columns:
            tmp = d0.groupby("participante_id", observed=True, as_index=False)[cov].mean()
            d = d.merge(tmp, on="participante_id", how="left")

    if d.empty:
        return None

    covars = ["IDATE_estado_pre"] + [c for c in use_covariates if c in d.columns]
    need = ["grupo", "IDATE_estado_pos"] + covars

    # Tipos + saneamento
    for c in need:
        if c != "grupo":
            d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        else:
            d[c] = d[c].astype("category").cat.remove_unused_categories()

    d2 = d.dropna(subset=need)

    # Requisitos mínimos
    if d2["grupo"].nunique() < 2 or (d2.groupby("grupo").size() < 2).any():
        return None

    # (Opcional) padronize covariáveis para melhorar condicionamento
    for c in covars:
        mu, sd = d2[c].mean(), d2[c].std(ddof=1)
        if not np.isfinite(sd) or sd == 0:
            return None
        d2[c] = (d2[c] - mu) / sd

    try:
        # >>> Correção principal: between deve ser string, não lista
        res = pg.ancova(data=d2, dv="IDATE_estado_pos", covar=covars, between="grupo")
        return res
    except Exception:
        # Fallback correto: ANOVA de SM por OLS (typ=2) + (opcional) eta parcial
        formula = "IDATE_estado_pos ~ C(grupo) + " + " + ".join(covars)
        try:
            ols = smf.ols(formula, data=d2).fit()
            aov = sm.stats.anova_lm(ols, typ=2)
            # (Opcional) eta parcial:
            ss_effects = aov["sum_sq"]
            ss_total = ss_effects.sum()
            aov["eta2_partial"] = aov["sum_sq"] / (aov["sum_sq"] + (ss_total - aov["sum_sq"]))
            return aov.reset_index().rename(columns={"index": "Source"})
        except Exception:
            return None

def h4_baseline_tests(df_participant: pd.DataFrame, which: str) -> Dict[str, Any]:
    out = {"measure": which}
    if which not in df_participant.columns or "grupo" not in df_participant.columns:
        out["error"] = "Colunas ausentes"
        return out
    d = (df_participant.groupby(["participante_id", "grupo"], observed=True, as_index=False)
                      .agg(val=(which, "mean")))
    groups = list(d["grupo"].unique())
    if len(groups) != 2:
        out["error"] = "São necessários 2 grupos."
        return out
    g1 = d[d["grupo"] == groups[0]]["val"].dropna().values
    g2 = d[d["grupo"] == groups[1]]["val"].dropna().values

    p1 = shapiro(g1)[1] if len(g1) >= 3 else 1.0
    p2 = shapiro(g2)[1] if len(g2) >= 3 else 1.0
    if p1 >= 0.05 and p2 >= 0.05:
        stat, p = ttest_ind(g1, g2, equal_var=False)
        s_denom = (len(g1)+len(g2)-2)
        s = math.sqrt(((len(g1)-1)*np.var(g1, ddof=1) + (len(g2)-1)*np.var(g2, ddof=1)) / s_denom) if s_denom > 0 else np.nan
        d_eff = (np.mean(g1) - np.mean(g2)) / s if s and np.isfinite(s) else np.nan
        out.update({"test": "t-Welch", "stat": float(stat), "p": float(p), "effect_d": float(d_eff)})
    else:
        stat, p = mannwhitneyu(g1, g2, alternative="two-sided")
        n = len(g1) + len(g2)
        z = (stat - (len(g1)*len(g2)/2)) / math.sqrt(len(g1)*len(g2)*(n+1)/12)
        r = z / math.sqrt(n)
        out.update({"test": "Mann-Whitney", "stat": float(stat), "p": float(p), "effect_r": float(r)})
    out.update({"n1": int(len(g1)), "n2": int(len(g2)),
                "mean1": float(np.mean(g1)) if len(g1) else np.nan,
                "mean2": float(np.mean(g2)) if len(g2) else np.nan})
    return out

# ---------------------------------------
# Plots
# ---------------------------------------

def plot_interaction(df: pd.DataFrame, dv: str = "bpm_metric") -> go.Figure:
    d = df.dropna(subset=[dv, "grupo", "fase"]).copy()
    agg = (d.groupby(["grupo", "fase"], observed=True, as_index=False)
             .agg(mean=(dv, "mean"),
                  sd=(dv, "std"),
                  n=(dv, "count")))
    agg["sem"] = agg["sd"] / np.sqrt(agg["n"].clip(lower=1))
    fig = px.line(agg, x="fase", y="mean", color="grupo", markers=True)
    for g in agg["grupo"].unique():
        sub = agg[agg["grupo"] == g]
        fig.add_trace(go.Scatter(x=sub["fase"], y=sub["mean"] + sub["sem"], mode="lines",
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=sub["fase"], y=sub["mean"] - sub["sem"], mode="lines",
                                 fill="tonexty", line=dict(width=0), name=f"{g} (±SEM)", hoverinfo="skip"))
    fig.update_layout(yaxis_title=f"{dv} (média ± SEM)", xaxis_title="Fase")
    return fig

def plot_idate_boxes(df_participant: pd.DataFrame) -> Tuple[Optional[go.Figure], Optional[go.Figure]]:
    need = {"grupo", "IDATE_estado_pre", "IDATE_estado_pos", "participante_id"}
    if not need.issubset(df_participant.columns):
        return None, None
    d = (df_participant.groupby(["participante_id", "grupo"], observed=True, as_index=False)
                      .agg(pre=("IDATE_estado_pre", "mean"),
                           pos=("IDATE_estado_pos", "mean")))
    fig1 = px.box(d, x="grupo", y="pre", points="all", title="IDATE Estado (Pré)")
    fig2 = px.box(d, x="grupo", y="pos", points="all", title="IDATE Estado (Pós)")
    return fig1, fig2

def plot_scatter_corr(df_participant: pd.DataFrame, physio: str, subj: str) -> Optional[go.Figure]:
    need = {"grupo", "participante_id", physio, subj}
    if not need.issubset(df_participant.columns):
        return None
    d = (df_participant.groupby(["participante_id", "grupo"], observed=True, as_index=False)
                      .agg(**{physio: (physio, "mean"), subj: (subj, "mean")}))
    fig = px.scatter(d, x=physio, y=subj, color="grupo", trendline="ols", hover_data=["participante_id"])
    fig.update_layout(title=f"Correlação {physio} × {subj} por grupo")
    return fig

# ---------------------------------------
# App
# ---------------------------------------

def main():
    st.sidebar.header("Configuração & Filtros")
    db_path = st.sidebar.text_input("Caminho do SQLite", value="psicologia.db")
    metric_key = st.sidebar.selectbox("Métrica de BPM", ["bpm_mean", "bpm_max", "bpm_peaks"], index=0)
    include_cardiopaths = st.sidebar.checkbox("Incluir cardiopatas", value=False)

    with st.sidebar.expander("❓ Ajuda rápida"):
        st.markdown(
            "- **Métrica de BPM**: escolha qual resumo usar nas análises (média, máximo, contagem de picos > Q95).  \n"
            "- **Cardiopatas**: por padrão são excluídos para reduzir viés fisiológico."
        )

    if not os.path.exists(db_path):
        st.error("Banco SQLite não encontrado. Ajuste o caminho na sidebar.")
        st.stop()

    tables = load_sqlite_tables(db_path)
    if not tables:
        st.error("Sem tabelas no SQLite.")
        st.stop()

    grupos = tables.get("participantes", pd.DataFrame()).get("grupo", pd.Series(dtype=str)).dropna().unique().tolist()
    selected_groups = st.sidebar.multiselect("Grupos", options=sorted(grupos), default=sorted(grupos))

    df, participantes = master_table(tables, metric_key, include_cardiopaths, selected_groups)

    # saneamento de 'grupo' e numéricos usados nos cálculos
    if "grupo" in df.columns:
        df["grupo"] = df["grupo"].astype("category").cat.remove_unused_categories()

    # seletor de participante
    pid_opts = df["participante_id"].dropna().unique().tolist()
    pid_sel = st.sidebar.selectbox("Participante (opcional p/ série bruta)", options=["<agregado>"] + pid_opts)

    cov_opts = [c for c in ["idade", "realismo", "enjoo"] if c in df.columns]
    use_covs = st.sidebar.multiselect("Covariáveis (ANCOVA)", options=cov_opts, default=[])

    st.title("VR Fobia a Aranhas — Análise")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Resumo Executivo",
        "Análise Fisiológica (BPM)",
        "Análise Psicométrica (Questionários)",
        "Correlações & Subjetivo",
        "Hipóteses de Pesquisa",
        "Glossário",
        "Referências"
    ])

    # ============== TAB 1: Resumo ==============
    with tab1:
        col1, col2, col3, col4 = st.columns(4)

        # H1 — modelos
        lme = fit_lme_h1(df, dv="bpm_metric")
        p_inter = np.nan
        if lme is not None:
            try:
                summary = lme.summary().tables[1]
                coef_df = pd.DataFrame(summary.data[1:], columns=summary.data[0])
                inter_rows = coef_df[coef_df["Coef."].astype(str).str.contains("C\\(grupo\\):C\\(fase_c\\)")]
                if not inter_rows.empty and "P>|z|" in inter_rows.columns:
                    p_inter = float(pd.to_numeric(inter_rows["P>|z|"], errors="coerce").min())
            except Exception:
                pass

        anova = anova_mista_h1(df, dv="bpm_metric")
        p_inter_anova = (anova.loc[anova["Source"] == "Interaction", "p-unc"].values[0]
                         if anova is not None and "Interaction" in anova["Source"].values else np.nan)
        np2_inter = (anova.loc[anova["Source"] == "Interaction", "np2"].values[0]
                     if anova is not None and "Interaction" in anova["Source"].values else np.nan)

        # H2 — correlações e comparação
        phys_default = "bpm_max"
        subj_default = "ansiedade" if "ansiedade" in df.columns else ("aversao_aranha" if "aversao_aranha" in df.columns else None)
        r_fob, n_fob, r_ctrl, n_ctrl, p_diff = (np.nan, 0, np.nan, 0, np.nan)
        if subj_default is not None:
            if pd.api.types.is_categorical_dtype(df["grupo"]):
                df["grupo"] = df["grupo"].cat.remove_unused_categories()
            df[phys_default] = pd.to_numeric(df[phys_default], errors="coerce")
            df[subj_default] = pd.to_numeric(df[subj_default], errors="coerce")

            agg = (
                df.dropna(subset=[phys_default, subj_default, "grupo", "participante_id"])
                  .groupby(["participante_id", "grupo"], observed=True, as_index=False)
                  .agg(**{phys_default: (phys_default, "mean"),
                          subj_default: (subj_default, "mean")})
            )
            if not agg.empty:
                if "fobico" in agg["grupo"].unique():
                    sub = agg[agg["grupo"] == "fobico"]
                    if len(sub) >= 3:
                        r_fob = float(pg.corr(sub[phys_default], sub[subj_default], method="pearson")["r"])
                        n_fob = int(len(sub))
                ctrl_labels = [g for g in agg["grupo"].unique() if g != "fobico"]
                if ctrl_labels:
                    sub = agg[agg["grupo"] == ctrl_labels[0]]
                    if len(sub) >= 3:
                        r_ctrl = float(pg.corr(sub[phys_default], sub[subj_default], method="pearson")["r"])
                        n_ctrl = int(len(sub))
                if n_fob >= 4 and n_ctrl >= 4 and np.isfinite(r_fob) and np.isfinite(r_ctrl):
                    p_diff = compare_correlations_fisher(r_fob, n_fob, r_ctrl, n_ctrl)["p"]

        col1.metric("H1 — p(Interação) LME", f"{p_inter:.3g}" if np.isfinite(p_inter) else "—")
        col2.metric("H1 — p(Interação) ANOVA", f"{p_inter_anova:.3g}" if np.isfinite(p_inter_anova) else "—")
        col3.metric("H1 — η² parcial", f"{np2_inter:.3f}" if np.isfinite(np2_inter) else "—")
        col4.metric("H2 — p(diff r_fóbico vs r_ctrl)", f"{p_diff:.3g}" if np.isfinite(p_diff) else "—")

        with st.expander("❓ O que é cada métrica do Resumo?"):
            st.markdown(
                "- **H1 — p(Interação) LME**: p-valor do termo **Grupo×Fase** no **modelo de efeitos mistos**. "
                "Indica se o padrão de mudança do BPM ao longo das fases difere entre os grupos.\n"
                "- **H1 — p(Interação) ANOVA**: mesmo conceito, mas pela **ANOVA mista** (boa para divulgação quando pressupostos são razoáveis).\n"
                "- **H1 — η² parcial**: **tamanho de efeito** da interação na ANOVA (proporção da variância do DV explicada pelo fator, **controlando** os demais). "
                "Regra prática: ~0.01 pequeno, ~0.06 médio, ~0.14 grande (Cohen).\n"
                "- **H2 — p(diff r)**: p-valor do **teste de Fisher r-to-z** que compara se a correlação (fisiologia×subjetivo) é diferente entre fóbicos e controle."
            )

        st.caption("Preferimos LME pela robustez; ANOVA mista exibida por transparência. H2 usa Fisher r-to-z.")

    # ============== TAB 2: BPM ==============
    with tab2:
        st.subheader("Interação Grupo × Fase")
        st.plotly_chart(plot_interaction(df, dv="bpm_metric"), use_container_width=True)

        with st.expander("❓ Interpretação do gráfico de interação"):
            st.markdown(
                "Linhas são a **média** da métrica de BPM por fase; a faixa sombreada é **±SEM**. "
                "Uma **divergência** clara entre as linhas (com mudança diferente ao longo das fases) sugere interação Grupo×Fase."
            )

        st.markdown("### Resultados estatísticos")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Modelo de Efeitos Mistos (LME)**")
            if lme is None:
                st.warning("LME não estimado (dados insuficientes ou falha numérica).")
            else:
                st.text(lme.summary())

        with c2:
            st.write("**ANOVA Mista (pingouin)**")
            if anova is None:
                st.warning("ANOVA mista não disponível.")
            else:
                st.dataframe(anova)

        if pid_sel != "<agregado>":
            st.markdown("### Série temporal (participante selecionado)")
            fases_long = unify_fases(tables)
            one = fases_long[fases_long["participante_id"] == pid_sel].copy()
            rows = []
            for _, r in one.iterrows():
                series = parse_session_bpm(r.get("session_bpm"))
                if series:
                    rows += [{"fase": int(r["fase"]), "t": i, "bpm": v} for i, v in enumerate(series)]
            if rows:
                raw = pd.DataFrame(rows)
                fig_raw = px.line(raw, x="t", y="bpm", color=raw["fase"].astype(str), title=f"Série BPM — Participante {pid_sel}")
                st.plotly_chart(fig_raw, use_container_width=True)
            else:
                st.info("Sem série temporal parsável para este participante.")

    # ============== TAB 3: Psicometria ==============
    with tab3:
        st.subheader("Distribuições IDATE (Pré/Pós)")
        if set(["IDATE_estado_pre", "IDATE_estado_pos"]).issubset(df.columns):
            byp = (df.groupby(["participante_id", "grupo"], observed=True, as_index=False)
                     .agg(IDATE_estado_pre=("IDATE_estado_pre", "mean"),
                          IDATE_estado_pos=("IDATE_estado_pos", "mean")))
            fig_pre = px.box(byp, x="grupo", y="IDATE_estado_pre", points="all", title="IDATE Estado (Pré)")
            fig_pos = px.box(byp, x="grupo", y="IDATE_estado_pos", points="all", title="IDATE Estado (Pós)")
            st.plotly_chart(fig_pre, use_container_width=True)
            st.plotly_chart(fig_pos, use_container_width=True)
        else:
            st.warning("Colunas de IDATE não disponíveis (verifique a tabela 'idate').")

        with st.expander("❓ O que é ANCOVA e quando usar?"):
            st.markdown(
                "**ANCOVA** modela o **Pós** ajustando pelo **Pré** (covariável), além do fator **Grupo**. "
                "Requisitos: linearidade entre Pré e Pós, homogeneidade das inclinações (a relação Pré→Pós não deve diferir entre grupos) e resíduos aproximadamente normais."
            )

        st.markdown("### H3 — ANCOVA (Pós ajustado pelo Pré)")
        # # ------- DEBUG -------- #
        # # Debug rápido: coloque isso pouco antes de ancova_h3(df, use_covs)
        # st.write("DEBUG/ANCOVA — colunas no df:", set(df.columns))
        # st.write("DEBUG/ANCOVA — grupos (brutos):", df["grupo"].dropna().astype(str).unique().tolist())

        # dbg = (df.groupby(["participante_id","grupo"], observed=True, as_index=False)
        #         .agg(pre=("IDATE_estado_pre","mean"),
        #             pos=("IDATE_estado_pos","mean"))
        #         .dropna(subset=["pre","pos"]))
        # st.write("DEBUG/ANCOVA — participantes válidos por grupo:",
        #         dbg.groupby("grupo").size().to_dict())

        # # Se quiser ver porque o pivot falhou:
        # idate = tables.get("idate", pd.DataFrame())
        # if not idate.empty and "tipo" in idate.columns:
        #     aux = idate["tipo"].astype(str).str.strip().str.lower().value_counts()
        #     st.write("DEBUG/IDATE — top tipos:", aux.head(10).to_dict())
        #     st.write("DEBUG/IDATE — contém 'estado_pós' com acento?:",
        #      any("pós" in t for t in aux.index))
        # # ------- DEBUG -------- #


        anc = ancova_h3(df, use_covariates=use_covs)
        if anc is None or (isinstance(anc, pd.DataFrame) and anc.empty):
            st.warning("ANCOVA não pôde ser estimada (verifique colunas necessárias).")
        else:
            st.dataframe(anc)

        st.markdown("### H4 — Linha de base entre grupos")
        colx, coly = st.columns(2)
        with colx:
            res_pre = h4_baseline_tests(df, "IDATE_estado_pre") if "IDATE_estado_pre" in df.columns else {"error": "IDATE_estado_pre ausente"}
            st.json(res_pre)
        with coly:
            which = "IDATE_traco" if "IDATE_traco" in df.columns else "IDATE_estado_pre"
            res_traco = h4_baseline_tests(df, which) if which in df.columns else {"error": f"{which} ausente"}
            st.json(res_traco)

    # ============== TAB 4: Correlações ==============
    with tab4:
        st.subheader("Correlação fisiologia × subjetivo")
        phys_choice = st.selectbox("Métrica fisiológica", ["bpm_max", "bpm_mean", "bpm_peaks"], index=0)
        subj_candidates = [c for c in ["ansiedade", "aversao_aranha"] if c in df.columns]
        if not subj_candidates:
            st.info("Sem métricas subjetivas no banco.")
        else:
            subj_choice = st.selectbox("Métrica subjetiva", subj_candidates, index=0)
            figc = plot_scatter_corr(df, phys_choice, subj_choice)
            if figc:
                st.plotly_chart(figc, use_container_width=True)

            agg = (
                df.dropna(subset=[phys_choice, subj_choice, "grupo", "participante_id"])
                  .groupby(["participante_id", "grupo"], observed=True, as_index=False)
                  .agg(**{phys_choice: (phys_choice, "mean"), subj_choice: (subj_choice, "mean")})
            )
            rows = []
            for g, sub in agg.groupby("grupo", observed=True):
                if len(sub) >= 3:
                    res = pg.corr(sub[phys_choice], sub[subj_choice], method="pearson")
                    rows.append({"grupo": g, "n": int(len(sub)), "r": float(res["r"]), "p": float(res["p-val"])})
            if rows:
                tbl = pd.DataFrame(rows)
                st.dataframe(tbl, hide_index=True)
                if len(tbl) >= 2:
                    r1, n1 = tbl.iloc[0]["r"], int(tbl.iloc[0]["n"])
                    r2, n2 = tbl.iloc[1]["r"], int(tbl.iloc[1]["n"])
                    cmp_ = compare_correlations_fisher(r1, n1, r2, n2)
                    with st.expander("❓ Como interpretar a comparação de correlações?"):
                        st.markdown(
                            f"Compara **r** entre grupos via **Fisher r-to-z**. "
                            f"Se **p < 0.05**, a força da relação difere entre os grupos."
                        )
                    st.info(f"Comparação de correlações (Fisher r-to-z): z={cmp_['z']:.3f}, p={cmp_['p']:.3g}")

    # ============== TAB 5: Hipóteses de Pesquisa ==============
    with tab5:
        st.header("🔬 Hipóteses da Pesquisa")
        st.markdown("Aqui detalhamos as quatro principais previsões que guiam este estudo.")
        with st.expander("**H1: Interação Grupo-Estímulo**"):
            st.markdown(
                """
                - **Hipótese:** O aumento da frequência cardíaca (BPM) ao longo das fases de exposição será **significativamente maior no grupo fóbico** em comparação com o grupo controle.
                - **Em outras palavras:** Esperamos que o coração dos participantes fóbicos acelere muito mais do que o dos controles à medida que as aranhas virtuais se tornam mais ameaçadoras. A "reação ao estímulo" deve ser diferente entre os grupos.
                """
            )

        with st.expander("**H2: Correlação Diferencial**"):
            st.markdown(
                """
                - **Hipótese:** A correlação positiva entre as métricas fisiológicas (picos de BPM) e as respostas subjetivas de medo (questionário pós-VR) será **mais forte e mais significativa no grupo fóbico**.
                - **Em outras palavras:** Para os fóbicos, quanto mais o coração acelerar, maior será o medo que eles *relatam* ter sentido. Essa ligação entre corpo e mente (reação fisiológica e percepção do medo) deve ser mais fraca ou inexistente no grupo controle.
                """
            )

        with st.expander("**H3: Impacto na Ansiedade**"):
            st.markdown(
                """
                - **Hipótese:** O aumento nos scores de ansiedade-estado (IDATE pré vs. pós-experimento) será **significativamente maior para o grupo fóbico**.
                - **Em outras palavras:** Após a experiência na VR, esperamos que o nível de ansiedade momentânea do grupo fóbico aumente muito mais do que o do grupo controle, quando comparamos com os níveis de ansiedade que eles tinham antes de começar.
                """
            )

        with st.expander("**H4: Linha de Base**"):
            st.markdown(
                """
                - **Hipótese:** O grupo fóbico já apresentará níveis de ansiedade (no IDATE) **significativamente mais elevados** que o grupo controle, mesmo **antes** do início da exposição.
                - **Em outras palavras:** Esta é uma verificação de segurança. Queremos confirmar que o grupo que classificamos como "fóbico" já é, em geral, mais ansioso do que o grupo controle. Isso valida a separação dos nossos grupos.
                """
            )

    with tab6:
        st.header("📖 Glossário: Descomplicando a Estatística")
        st.markdown("Um guia rápido para entender os testes e métricas utilizados neste dashboard.")
        with st.expander("p-valor (valor-p)"):
            st.markdown(
                """
                - É Uma medida de **evidência estatística**. Representa a probabilidade de observermos os dados que coletamos (ou dados ainda mais extremos) se a nossa hipótese inicial ("não há efeito ou diferença") fosse verdadeira.
                - **Como interpretar?** Por convenção, se o **p-valor < 0.05**, consideramos o resultado "estatisticamente significativo". Isso significa que é improvável que a diferença que encontramos seja fruto do mero acaso, e podemos rejeitar a ideia de que "não há efeito".
                """
            )

        with st.expander("ANOVA Mista (Mixed ANOVA)"):
            st.markdown(
                """
                - É um teste que compara médias de grupos em diferentes momentos. É "mista" porque mistura uma comparação **entre** grupos diferentes (fóbico vs. controle) com uma comparação **dentro** dos mesmos indivíduos ao longo do tempo (fase 1 vs. fase 2 vs. fase 3).
                - **Por que usamos?** É o teste ideal para a **H1**, para verificar se a "trajetória" do BPM ao longo das fases é diferente para cada grupo. O resultado mais importante é o da **"Interação"**.
                """
            )

        with st.expander("ANCOVA (Análise de Covariância)"):
            st.markdown(
                """
                - É uma "versão inteligente" da ANOVA. Ela compara as médias dos grupos (ex: score de ansiedade *pós*-experimento) depois de **ajustar** estatisticamente o resultado com base em uma outra variável (a covariável).
                - **Por que usamos?** Para a **H3**. Comparamos a ansiedade *pós*-VR entre os grupos, mas usando a ansiedade *pré*-VR como covariável. Isso nos permite ver o efeito do experimento de forma mais "pura", controlando as diferenças de ansiedade que os participantes já tinham antes de começar.
                """
            )

        with st.expander("Teste t de Amostras Independentes e Teste de Mann-Whitney U"):
            st.markdown(
                """
                - Ambos são testes simples para comparar duas médias de dois grupos independentes (ex: fóbico vs. controle) em uma única medida.
                - **Qual a diferença?** O **Teste t** assume que os dados seguem uma distribuição normal. O **Teste de Mann-Whitney U** é a alternativa "não paramétrica", usada quando essa suposição não é atendida. O script escolhe automaticamente o mais adequado.
                - **Por que usamos?** Para a **H4**, para comparar os scores do IDATE entre os grupos *antes* do experimento começar.
                """
            )

        with st.expander("Correlação de Pearson (r)"):
            st.markdown(
                """
                - É uma medida que quantifica a força e a direção de uma relação *linear* entre duas variáveis contínuas. O valor `r` varia de -1 a +1.
                - **Como interpretar?** `r` perto de +1 indica uma forte relação positiva (quando um sobe, o outro sobe). `r` perto de -1 indica uma forte relação negativa (quando um sobe, o outro desce). `r` perto de 0 indica ausência de relação linear.
                - **Por que usamos?** Para a **H2**, para medir a associação entre o BPM e o medo reportado.
                """
            )

        with st.expander("Teste de Fisher r-to-z (p(diff r))"):
            st.markdown(
                """
                - **O que é?** Um teste estatístico usado para responder a uma pergunta específica: "A força da correlação (o valor `r`) no grupo A é significativamente diferente da força da correlação no grupo B?".
                - **Por que usamos?** É o passo final da **H2**. Depois de calcular a correlação para o grupo fóbico e para o controle, usamos este teste para provar que a correlação no grupo fóbico é, de fato, estatisticamente mais forte.
                """
            )

        with st.expander("Tamanho de Efeito (η² parcial e Cohen's d)"):
            st.markdown(
                """
                - Enquanto o p-valor nos diz se um efeito *existe* (se é significativo), o tamanho de efeito nos diz o quão *grande* ou *importante* é esse efeito.
                - **Como interpretar?**
                    - **Eta parcial ao quadrado (η² parcial)**: Usado na ANOVA/ANCOVA. Indica a proporção da variância explicada por um fator. Valores comuns: 0.01 (pequeno), 0.06 (médio), 0.14 (grande).
                    - **d de Cohen**: Usado no Teste t. Mede a diferença entre duas médias em termos de desvios-padrão. Valores comuns: 0.2 (pequeno), 0.5 (médio), 0.8 (grande).
                """
            )

        with st.expander("LME (Modelo de Efeitos Mistos)"):
            st.markdown(
                """
                - Trata-se modelo estatístico avançado, considerado uma alternativa mais robusta e flexível à ANOVA de medidas repetidas.
                - **Por que usamos?** Para a **H1**. Ele é excelente para dados longitudinais (medidas repetidas ao longo do tempo) e lida melhor com dados faltantes e estruturas de dados complexas do que a ANOVA tradicional. É por isso que o dashboard o apresenta como a análise preferencial.
                """
            )
    # ============== TAB 7: Referências  ==============
    with tab7:
        st.header("📚 Referências Científicas")
        st.markdown(
            """
            Esta seção lista as referências científicas que fundamentam as principais 
            escolhas metodológicas e estatísticas deste dashboard.
            """
        )
        st.subheader("Modelos de Efeitos Mistos (LME)")
        st.markdown(
            """
            - **Baayen, R. H., Davidson, D. J., & Bates, D. M. (2008).** Mixed-effects modeling with crossed random effects for subjects and items. *Journal of Memory and Language, 59*(4), 390–412.
            - **Winter, B. (2013).** Linear models and linear mixed effects models in R: A practical guide for biologists and psychologists. *arXiv preprint arXiv:1308.5499*.
            """
        )

        st.subheader("ANOVA e ANCOVA")
        st.markdown(
            """
            - **Field, A. (2018).** *Discovering Statistics Using IBM SPSS Statistics* (5th ed.). Sage Publications.
            """
        )

        st.subheader("Testes para Comparações de Grupos e Correlações")
        st.markdown(
            """
            - **Mann, H. B., & Whitney, D. R. (1947).** On a test of whether one of two random variables is stochastically larger than the other. *The Annals of Mathematical Statistics, 18*(1), 50–60.
            - **Steiger, J. H. (1980).** Tests for comparing elements of a correlation matrix. *Psychological Bulletin, 87*(2), 245–251.
            """
        )

        st.subheader("Tamanhos de Efeito e Teste de Normalidade")
        st.markdown(
            """
            - **Cohen, J. (1988).** *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.
            - **Lakens, D. (2013).** Calculating and reporting effect sizes to facilitate cumulative science: a practical primer for t-tests and ANOVAs. *Frontiers in Psychology, 4*, 863.
            - **Shapiro, S. S., & Wilk, M. B. (1965).** An analysis of variance test for normality (complete samples). *Biometrika, 52*(3/4), 591–611. [Leia aqui](https://www.jstor.org/stable/2333709?read-now=1&seq=1#page_scan_tab_contents)
            """
        )
        st.write("Google Drive com os artigos [https://drive.google.com/drive/folders/1ynkD3ukUouw9MYw8TCV8fLKOiJqT4M2y?usp=sharing]")

if __name__ == "__main__":
    main()

# seed_vr_fobia.py
# -*- coding: utf-8 -*-
"""
Populador de dados sintéticos para estudo VR Fobia a Aranhas.

Compatível com o schema definido em setup_bd.py:
- idate.tipo ∈ {'idate_traco','idate_estado_pre','idate_estado_pos'}
- pos_vr_questionario.{ansiedade, aversao_aranha, enjoo, realismo} ∈ [0..10] (INTEGER)
- fase1, fase2, fase3 com session_bpm TEXT (JSON), tempos REAL; fase3 com colunas tempo_encontrar_aranha_1..15 e tempo_encontrar_5a_aranha (compat).
"""

import sqlite3
import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import json
import numpy as np

# ===== Config =====
DB_PATH = "psicologia.db"
N_PARTICIPANTS = 120     # total
PROP_PHOBIC = 0.5        # ~50% fóbicos
SEED = 42

rng = np.random.default_rng(SEED)
random.seed(SEED)

# ===== Utils =====

def execmany(conn: sqlite3.Connection, sql: str, rows: List[Tuple[Any, ...]]):
    if not rows:
        return
    conn.executemany(sql, rows)

def table_cols(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.execute(f"PRAGMA table_info('{table}')")
    return [r[1] for r in cur.fetchall()]

def clamp(val: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, val)))

def likert(mean: float, sd: float = 0.8, lo: int = 1, hi: int = 4) -> int:
    """Gera resposta 1..4 (IDATE)."""
    v = int(round(clamp(rng.normal(mean, sd), lo, hi)))
    return v

def rand_phone() -> str:
    ddd = rng.choice([11, 21, 31, 41, 51, 61, 62, 71, 81, 85, 91])
    nine = rng.integers(90000, 99999)
    four = rng.integers(1000, 9999)
    return f"+55 ({ddd}) 9{nine}-{four:04d}"

def rand_name_gender() -> Tuple[str, str]:
    masc = ["João", "Pedro", "Lucas", "Gabriel", "Rafael", "Gustavo", "Daniel", "Felipe", "Thiago", "Bruno"]
    fem  = ["Maria", "Ana", "Beatriz", "Julia", "Mariana", "Carolina", "Camila", "Larissa", "Fernanda", "Patrícia"]
    sob  = ["Silva", "Souza", "Oliveira", "Santos", "Pereira", "Almeida", "Rodrigues", "Gomes", "Martins", "Barbosa"]
    if rng.random() < 0.5:
        n = f"{rng.choice(masc)} {rng.choice(sob)}"
        return n, "masculino"
    else:
        n = f"{rng.choice(fem)} {rng.choice(sob)}"
        return n, "feminino"

def date_cadastro() -> str:
    start = datetime.now() - timedelta(days=120)
    d = start + timedelta(days=int(rng.integers(0, 120)))
    return d.strftime("%Y-%m-%d")

# ===== Geração de participantes =====

def gen_participantes(n_total: int) -> List[Dict[str, Any]]:
    n_phobic = int(round(n_total * PROP_PHOBIC))
    rows: List[Dict[str,Any]] = []

    for i in range(n_total):
        nome, genero = rand_name_gender()
        grupo = "fobico" if i < n_phobic else "controle"
        idade = int(clamp(rng.normal(26 if grupo=="fobico" else 28, 5), 18, 55))
        email = (nome.lower().replace(" ", ".") + f"{rng.integers(1,999)}@exemplo.com")
        telefone = rand_phone()

        # SPQ 0..26
        spq_mean = 18 if grupo=="fobico" else 6
        pontuacao_spq = int(clamp(rng.normal(spq_mean, 4), 0, 26))

        usa_medicacao = int(rng.random() < (0.25 if grupo=="fobico" else 0.15))
        medicacao = "ansiolítico leve" if usa_medicacao and (rng.random()<0.6) else ("outros" if usa_medicacao else None)

        problema_cardiaco = int(rng.random() < 0.05)
        tontura = int(rng.random() < 0.10)
        disponibilidade = rng.choice(["manhã","tarde","noite"])

        rows.append(dict(
            nome=nome, email=email, telefone=telefone, idade=idade, genero=genero,
            grupo=grupo, usa_medicacao=usa_medicacao, medicacao=medicacao,
            problema_cardiaco=problema_cardiaco, tontura_nausea_migranea=tontura,
            disponibilidade=disponibilidade, pontuacao_spq=pontuacao_spq,
            data_cadastro=date_cadastro()
        ))
    return rows

# ===== Geração de IDATE =====

IDATE_ITEMS = 20  # por escala

def gen_idate(participantes: List[Tuple[int, str]]) -> List[Tuple[int, str, int, int]]:
    """
    Retorna linhas: (participante_id, tipo, questao, resposta)
    tipo ∈ {'idate_traco','idate_estado_pre','idate_estado_pos'}
    """
    out: List[Tuple[int,str,int,int]] = []
    for pid, grupo in participantes:
        # Traço (estável; fóbicos maiores)
        trait_mean = 3.0 if grupo=="fobico" else 2.2
        # Estado pré (baseline)
        state_pre_mean = 2.9 if grupo=="fobico" else 2.0
        # Estado pós (exposição ↑ ansiedade, mais forte nos fóbicos)
        delta = rng.normal(0.5 if grupo=="fobico" else 0.2, 0.25)
        state_pos_mean = state_pre_mean + max(0.0, delta)

        for q in range(1, IDATE_ITEMS+1):
            out.append((pid, "idate_traco", q, likert(trait_mean)))
            out.append((pid, "idate_estado_pre", q, likert(state_pre_mean)))
            out.append((pid, "idate_estado_pos", q, likert(state_pos_mean)))
    return out

# ===== Questionário pós-VR =====

def gen_pos_vr(participantes: List[Tuple[int, str]]) -> List[Tuple[int, int, int, int, int]]:
    """
    ansiedade, aversao_aranha, enjoo, realismo -> INTEGER 0..10
    """
    rows = []
    for pid, grupo in participantes:
        base_anx = rng.normal(7.5, 1.0) if grupo=="fobico" else rng.normal(4.0, 1.2)
        base_avs = rng.normal(8.0, 1.0) if grupo=="fobico" else rng.normal(3.5, 1.5)
        enjoo = clamp(rng.normal(3.0, 2.0), 0, 10)
        realismo = clamp(rng.normal(8.0, 1.0), 5, 10)
        rows.append((
            pid,
            int(round(clamp(base_anx, 0, 10))),
            int(round(clamp(base_avs, 0, 10))),
            int(round(enjoo)),
            int(round(realismo)),
        ))
    return rows

# ===== Séries de BPM por fase =====

def synth_bpm_stream(grupo: str, fase: int, dur_s: int) -> List[int]:
    base = rng.normal(75, 3) if grupo=="controle" else rng.normal(84, 3.5)
    inc = {1: rng.normal(3 if grupo=="controle" else 6, 1.0),
           2: rng.normal(5 if grupo=="controle" else 10, 1.5),
           3: rng.normal(6 if grupo=="controle" else 13, 2.0)}[fase]

    series = []
    level = base + inc
    t = 0
    while t < dur_s:
        breath = 1.5 * math.sin(2*math.pi*(t/5.5))
        drift = 0.003 * t
        noise = rng.normal(0, 1.2)
        val = level + breath + drift + noise

        if rng.random() < (0.015 if grupo=="controle" else 0.04) * (1 + 0.3*(fase-1)):
            peak = rng.normal(20 if grupo=="controle" else 35, 6)
            tau = int(rng.integers(3, 7))
            for k in range(t, min(t+tau, dur_s)):
                series.append(int(round(val + peak * math.exp(-(k-t)/max(1, tau)))))
            t += tau
            continue

        series.append(int(round(val)))
        t += 1

    series = [int(clamp(v, 45, 200)) for v in series]
    return series

def tempo_encontrar_uma_aranha(grupo: str, fase: int) -> int:
    base = {1: (35, 12), 2: (28, 10), 3: (20, 9)}[fase]
    mu = base[0] + (5 if grupo=="fobico" else 0)
    val = int(clamp(rng.normal(mu, base[1]), 5, 180))
    return val

def tempos_fase3_multiplas_aranhas(grupo: str, n: int) -> List[int]:
    times: List[int] = []
    for i in range(1, n+1):
        if rng.random() < (0.12 if grupo=="fobico" else 0.05):
            times.append(None)  # não encontrou
        else:
            base = 12 + 0.8*i
            jitter = rng.normal(0, 4)
            penalty = 4 if (grupo=="fobico" and rng.random()<0.3) else 0
            t = int(clamp(base + jitter + penalty, 3, 180))
            times.append(t)
    return times

# ===== Inserção =====

def truncate_if_any(conn: sqlite3.Connection, table: str):
    try:
        conn.execute(f"DELETE FROM {table}")
    except sqlite3.Error:
        pass

def insert_participantes(conn: sqlite3.Connection, rows: List[Dict[str,Any]]) -> List[Tuple[int, str]]:
    cols = table_cols(conn, "participantes")
    fields = [c for c in [
        "nome","email","telefone","idade","genero","grupo","usa_medicacao","medicacao",
        "problema_cardiaco","tontura_nausea_migranea","disponibilidade",
        "pontuacao_spq","data_cadastro"
    ] if c in cols]

    sql = f"INSERT INTO participantes ({', '.join(fields)}) VALUES ({', '.join(['?']*len(fields))})"
    ids = []
    for r in rows:
        conn.execute(sql, tuple(r.get(f) for f in fields))
        pid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        ids.append(pid)
    groups = [r["grupo"] for r in rows]
    return list(zip(ids, groups))

def insert_idate(conn: sqlite3.Connection, rows: List[Tuple[int,str,int,int]]):
    cols = table_cols(conn, "idate")
    need = {"participante_id","tipo","questao","resposta"}
    if not need.issubset(set(cols)):
        return
    sql = "INSERT INTO idate (participante_id, tipo, questao, resposta) VALUES (?, ?, ?, ?)"
    execmany(conn, sql, rows)

def insert_pos_vr(conn: sqlite3.Connection, rows: List[Tuple[int,int,int,int,int]]):
    cols = table_cols(conn, "pos_vr_questionario")
    fields = [c for c in ["participante_id","ansiedade","aversao_aranha","enjoo","realismo"] if c in cols]
    sql = f"INSERT INTO pos_vr_questionario ({', '.join(fields)}) VALUES ({', '.join(['?']*len(fields))})"
    execmany(conn, sql, [tuple(r[:len(fields)]) for r in rows])

def insert_fase_simple(conn: sqlite3.Connection, table: str, rows: List[Dict[str,Any]]):
    cols = table_cols(conn, table)
    fields = [c for c in ["participante_id","session_bpm","tempo_finalizacao","tempo_encontrar_aranha"] if c in cols]
    if not fields:
        return
    sql = f"INSERT INTO {table} ({', '.join(fields)}) VALUES ({', '.join(['?']*len(fields))})"
    data = [tuple(r.get(f) for f in fields) for r in rows]
    execmany(conn, sql, data)

def insert_fase3(conn: sqlite3.Connection, rows: List[Dict[str,Any]]):
    cols = table_cols(conn, "fase3")
    time_cols = [c for c in cols if c.startswith("tempo_encontrar_aranha_")]
    if "tempo_encontrar_5a_aranha" in cols and "tempo_encontrar_aranha_5" not in time_cols:
        time_cols.append("tempo_encontrar_5a_aranha")
    base_fields = [c for c in ["participante_id","session_bpm","tempo_finalizacao","total_aranhas_encontradas"] if c in cols]
    fields = base_fields + sorted(time_cols, key=lambda s: (len(s), s))
    sql = f"INSERT INTO fase3 ({', '.join(fields)}) VALUES ({', '.join(['?']*len(fields))})"

    data = []
    for r in rows:
        row = [r.get(f) for f in base_fields]
        for tc in fields[len(base_fields):]:
            row.append(r.get(tc))
        data.append(tuple(row))
    execmany(conn, sql, data)

# ===== Pipeline =====

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")

    # Limpa (se existirem)
    for t in ["idate","pos_vr_questionario","fase1","fase2","fase3","participantes"]:
        truncate_if_any(conn, t)

    # Participantes
    parts_meta = gen_participantes(N_PARTICIPANTS)
    parts = insert_participantes(conn, parts_meta)  # List[(id, grupo)]

    # IDATE
    idate_rows = gen_idate(parts)
    insert_idate(conn, idate_rows)

    # Pós-VR (inteiros)
    pos_rows = gen_pos_vr(parts)
    insert_pos_vr(conn, pos_rows)

    # Fases 1 e 2
    fase1_rows = []
    fase2_rows = []
    for pid, grupo in parts:
        dur1 = int(rng.integers(220, 330))   # ~4–5.5 min
        dur2 = int(rng.integers(240, 360))   # ~4–6 min
        bpm1 = synth_bpm_stream(grupo, 1, dur1)
        bpm2 = synth_bpm_stream(grupo, 2, dur2)
        fase1_rows.append(dict(
            participante_id=pid,
            session_bpm=json.dumps(bpm1, ensure_ascii=False),
            tempo_finalizacao=dur1,
            tempo_encontrar_aranha=tempo_encontrar_uma_aranha(grupo, 1)
        ))
        fase2_rows.append(dict(
            participante_id=pid,
            session_bpm=json.dumps(bpm2, ensure_ascii=False),
            tempo_finalizacao=dur2,
            tempo_encontrar_aranha=tempo_encontrar_uma_aranha(grupo, 2)
        ))
    insert_fase_simple(conn, "fase1", fase1_rows)
    insert_fase_simple(conn, "fase2", fase2_rows)

    # Fase 3
    fase3_cols = table_cols(conn, "fase3")
    n_aranhas = max([int(c.split("_")[-1]) for c in fase3_cols if c.startswith("tempo_encontrar_aranha_")] + [15])
    fase3_rows = []
    for pid, grupo in parts:
        dur3 = int(rng.integers(260, 420))  # ~4.5–7 min
        bpm3 = synth_bpm_stream(grupo, 3, dur3)
        tempos = tempos_fase3_multiplas_aranhas(grupo, n_aranhas)
        total = int(sum(1 for t in tempos if t is not None))
        row = dict(
            participante_id=pid,
            session_bpm=json.dumps(bpm3, ensure_ascii=False),
            tempo_finalizacao=dur3,
            total_aranhas_encontradas=total
        )
        for i, t in enumerate(tempos, start=1):
            col_std = f"tempo_encontrar_aranha_{i}"
            col_alt = "tempo_encontrar_5a_aranha" if i == 5 else None
            if col_std in fase3_cols:
                row[col_std] = t
            elif col_alt and col_alt in fase3_cols:
                row[col_alt] = t
        fase3_rows.append(row)
    insert_fase3(conn, fase3_rows)

    conn.commit()
    conn.close()
    print(f"[OK] Banco populado em '{DB_PATH}' com {len(parts)} participantes.")

if __name__ == "__main__":
    main()

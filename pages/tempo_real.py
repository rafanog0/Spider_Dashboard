import streamlit as st
import sqlite3
import time
from datetime import datetime, timezone
import plotly.graph_objs as go
from func.back_to_home import render_back_to_home_button
from func.salva_cortado import extract_bpm  # usar se desejar leitura real
from func.vr_utils import detectar_vr, extrair_json, process_and_save
import random
import json
import os
from unidecode import unidecode

# Configurações do banco e paths
DB_PATH = "psicologia.db"
VR_SRC_PATH = "/media/oculos/SessionData/resultado.json"


def buscar_participantes():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, nome, idade, grupo, pontuacao_spq,
               problema_cardiaco, usa_medicacao, tontura_nausea_migranea
        FROM participantes ORDER BY nome
    """)
    resultados = cursor.fetchall()
    conn.close()
    return [{
        "id": r[0],
        "nome": r[1],
        "idade": r[2],
        "grupo": r[3],
        "pontuacao_spq": r[4],
        "problema_cardiaco": r[5],
        "usa_medicacao": r[6],
        "tontura_nausea_migranea": r[7]
    } for r in resultados]


def visualizar_participante():
    render_back_to_home_button()
    st.title("Sessão de Monitoramento em Tempo Real 🫀")

    # Inicializa estado
    defaults = {
        "sessao_ativa": False,
        "participante": None,
        "inicio_sessao": None,
        "bpm_dados": [],
        "timestamps": []
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Fluxo antes de iniciar
    if not st.session_state.sessao_ativa:
        participantes = buscar_participantes()
        if not participantes:
            st.warning("Nenhum participante encontrado.")
            return

        nome_to_participante = {p["nome"]: p for p in participantes}
        nome_escolhido = st.selectbox(
            "Selecione o participante", list(nome_to_participante.keys()),
            key="select_participante"
        )
        participante = nome_to_participante[nome_escolhido]

        if st.button("▶️ Iniciar Sessão", help='Inicie a sessão antes de iniciar o VR'):
            st.session_state.sessao_ativa = True
            st.session_state.participante = participante
            st.session_state.inicio_sessao = datetime.now(timezone.utc)
            st.session_state.bpm_dados = []
            st.session_state.timestamps = []
            st.rerun()

        # Exibe informações do participante
        with st.container():
            st.markdown("<!-- seu HTML/CSS de info-card aqui -->", unsafe_allow_html=True)

    # Fluxo durante sessão ativa
    else:
        participante = st.session_state.participante
        inicio_sessao = st.session_state.inicio_sessao

        st.success("Sessão iniciada!")
        st.markdown(f"### Participante: {participante['nome']} ({participante['grupo']})")
        st.markdown(f"**Início da Sessão:** `{inicio_sessao.strftime('%Y-%m-%d %H:%M:%S')}`")

        col1, col2 = st.columns([2, 1])
        bpm_placeholder = col1.empty()
        grafico_placeholder = col1.empty()
        fig = go.Figure()

        # Botão de encerramento
        if col2.button("⏹️ Encerrar Sessão"):
            fim_sessao = datetime.now()
            st.session_state.sessao_ativa = False

            # Salva JSON da dashboard
            dados_sessao = {
                "participante": participante,
                "inicio_sessao": inicio_sessao.strftime('%Y-%m-%d %H:%M:%S'),
                "fim_sessao": fim_sessao.strftime('%Y-%m-%d %H:%M:%S'),
                "bpm_dados": st.session_state.bpm_dados,
                "timestamps": st.session_state.timestamps,
            }
            os.makedirs("dados_sessoes", exist_ok=True)
            nome_limpo = unidecode(participante["nome"]).replace(" ", "_")
            dash_path = f"dados_sessoes/sessao_{nome_limpo}_dashboard.json"
            with open(dash_path, "w", encoding="utf-8") as f:
                json.dump(dados_sessao, f, ensure_ascii=False, indent=4)
            st.success(f"Dashboard salvo em: `{dash_path}`")

            # Aguarda conexão VR e extrai JSON
            vr_placeholder = st.empty()
            vr_placeholder.info("Aguardando conexão do Oculus VR…")
            detectar_vr()  # bloqueia até headset conectado
            vr_path = extrair_json(nome_limpo)
            vr_placeholder.success(f"Headset VR conectado: JSON salvo em `{vr_path}`")

            # Processa e salva no banco
            process_and_save(DB_PATH, dash_path, vr_path, participante['id'])
            st.success("Dados processados e armazenados no banco de dados.")

            st.rerun()

        bpm_atual = extract_bpm()
        #bpm_atual = random.randint(80, 140)
        horario_atual = datetime.now().strftime('%H:%M:%S')
        st.session_state.bpm_dados.append(bpm_atual)
        st.session_state.timestamps.append(horario_atual)

        cor = "green" if bpm_atual < 75 else "orange" if bpm_atual < 110 else "red"
        bpm_placeholder.markdown(
            f"<div style='text-align: center; font-size: 50px; font-weight: bold; color: {cor};'>{bpm_atual} BPM</div>",
            unsafe_allow_html=True
        )

        # Atualiza gráfico
        tempo = list(range(len(st.session_state.bpm_dados)))
        fig.add_trace(go.Scatter(
            x=tempo,
            y=st.session_state.bpm_dados,
            mode="lines+markers",
            line=dict(color="cyan")
        ))
        fig.update_layout(
            title="Frequência Cardíaca (BPM)",
            xaxis_title="Tempo (s)",
            yaxis_title="BPM",
            yaxis=dict(range=[50, 180]),
            template="plotly_dark",
            margin=dict(l=40, r=40, t=40, b=40),
            height=300,
        )
        grafico_placeholder.plotly_chart(fig, use_container_width=True)

        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    visualizar_participante()

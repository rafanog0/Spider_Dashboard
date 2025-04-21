import streamlit as st
import sqlite3
import time
from datetime import datetime
import plotly.graph_objs as go
from func.back_to_home import render_back_to_home_button
from func.salva_cortado import extract_bpm
import random

DB_PATH = "psicologia.db"

def buscar_participantes():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, nome, idade, grupo FROM participantes ORDER BY nome")
    resultados = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "nome": r[1], "idade": r[2], "grupo": r[3]} for r in resultados]

def visualizar_participante():
    render_back_to_home_button()
    st.title("Sessão de Monitoramento em Tempo Real 🫀")

    participantes = buscar_participantes()
    if not participantes:
        st.warning("Nenhum participante encontrado.")
        return

    nome_to_participante = {p["nome"]: p for p in participantes}
    nome_escolhido = st.selectbox("Selecione o participante", list(nome_to_participante.keys()))
    participante = nome_to_participante[nome_escolhido]

    tarefa = st.selectbox("Selecione a fase da tarefa", ["fase1", "fase2", "fase3"])

    if st.button("▶️ Iniciar Sessão"):
        st.success("Sessão iniciada!")
        inicio_sessao = datetime.now()
        st.markdown(f"### Participante: {participante['nome']} ({participante['grupo']})")
        st.markdown(f"**Início da Sessão:** `{inicio_sessao.strftime('%Y-%m-%d %H:%M:%S')}`")
        st.markdown(f"**Tarefa selecionada:** `{tarefa}`")

        col1, col2 = st.columns([2, 1])
        bpm_placeholder = col1.empty()
        grafico_placeholder = col1.empty()

        bpm_dados = []
        tempo = []
        fig = go.Figure()
        fig.update_layout(
            title="Frequência Cardíaca (BPM)",
            xaxis_title="Tempo (s)",
            yaxis_title="BPM",
            yaxis=dict(range=[50, 180]),
            template="plotly_dark",
            margin=dict(l=40, r=40, t=40, b=40),
            height=300,
        )

        i = 0
        rodando = True
        str_bpm = ''
        # Botão de parada fora do loop, dentro do layout
        with col2:
            if st.button("⏹️ Encerrar Sessão"):
                st.warning("Clique em Iniciar novamente para uma nova sessão.")
                rodando = False

        # Loop principal de leitura
        while rodando:
            # bpm_atual = int(extract_bpm())
            bpm_atual = random.randint(80, 140)
            hrv = round(60 / bpm_atual + (0.1 * (0.5 - time.time() % 1)), 2)  # simulado
            date_string = str(datetime.now().strftime('%m-%d %H:%M:%S'))
            str_bpm += f'{bpm_atual}_{date_string}'
            bpm_dados.append(bpm_atual)
                
            tempo.append(i)

            cor = "green" if bpm_atual < 75 else "orange" if bpm_atual < 110 else "red"
            bpm_placeholder.markdown(
                f"<div style='text-align: center; font-size: 50px; font-weight: bold; color: {cor};'>{bpm_atual} BPM</div>",
                unsafe_allow_html=True
            )

            fig.data = []
            fig.add_trace(go.Scatter(x=tempo, y=bpm_dados, mode="lines+markers", line=dict(color="cyan")))
            grafico_placeholder.plotly_chart(fig, use_container_width=True)

            time.sleep(1)
            i += 1

            # Permitir o encerramento ao clicar fora do botão
            if st.session_state.get("stop_session"):
                st.success("Sessão encerrada.")
                break

if __name__ == "__main__":
    visualizar_participante()

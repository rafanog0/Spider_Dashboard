import streamlit as st
import sqlite3
from func.back_to_home import render_back_to_home_button

DB_PATH = "psicologia.db"

def buscar_participantes():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, nome FROM participantes ORDER BY nome")
    participantes = cursor.fetchall()
    conn.close()
    return participantes

def salvar_idate(participante_id, tipo, respostas):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Remove registros anteriores desse tipo para o mesmo participante (evita duplicação)
    cursor.execute("""
        DELETE FROM idate
        WHERE participante_id = ? AND tipo = ?
    """, (participante_id, tipo))

    # Insere as novas respostas
    for i, resposta in enumerate(respostas, start=1):
        cursor.execute("""
            INSERT INTO idate (participante_id, tipo, questao, resposta)
            VALUES (?, ?, ?, ?)
        """, (participante_id, tipo, i, resposta))
    conn.commit()
    conn.close()

def idate_form():
    render_back_to_home_button()
    st.title("Formulário IDATE (Inventário de Ansiedade)")

    participantes = buscar_participantes()
    if not participantes:
        st.warning("Nenhum participante cadastrado.")
        return

    nome_to_id = {nome: id_ for id_, nome in participantes}
    nome_selecionado = st.selectbox("Selecione o participante", list(nome_to_id.keys()))
    participante_id = nome_to_id[nome_selecionado]

    tipo_idate = st.radio("Tipo do questionário", ["estado", "traço"])

    st.markdown("### Responda às 20 questões da escala:")
    with st.form("form_idate"):
        respostas = []
        for i in range(1, 21):
            resposta = st.slider(f"Questão {i}", 1, 4, value=2)
            respostas.append(resposta)

        if st.form_submit_button("Salvar respostas"):
            salvar_idate(participante_id, tipo_idate, respostas)
            st.success("Respostas do IDATE salvas com sucesso!")

if __name__ == "__main__":
    idate_form()

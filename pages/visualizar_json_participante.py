import os
import json
import sqlite3
import streamlit as st
from func.back_to_home import render_back_to_home_button

# Configuração da página
st.set_page_config(
    page_title="JSON do Participante",
    page_icon="📄",
    layout="wide",
)

def get_db_path():
    # Caminho para o banco de dados na raiz do projeto
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "psicologia.db"))


def buscar_participantes():
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute("SELECT id, nome FROM participantes ORDER BY nome")
    participantes = cursor.fetchall()
    conn.close()
    return participantes


def main():
    render_back_to_home_button()
    st.title("📄 Visualizar JSON por Participante")

    participantes = buscar_participantes()
    if not participantes:
        st.warning("Nenhum participante cadastrado.")
        return

    # Seleção de participante
    nomes = [p[1] for p in participantes]
    nome_selecionado = st.selectbox("Selecione o participante", nomes)
    participante_id = next(p[0] for p in participantes if p[1] == nome_selecionado)

    st.markdown("### Cole abaixo o JSON de cenas para este participante:")
    json_input = st.text_area("JSON de Cenas", height=200)

    if json_input:
        try:
            data = json.loads(json_input)
            st.subheader("Dados JSON")
            st.json(data)
        except json.JSONDecodeError:
            st.error("JSON inválido. Verifique a formatação.")
    else:
        st.info("Aguardando JSON.")

if __name__ == "__main__":
    main()

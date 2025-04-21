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

def salvar_questionario(participante_id, dados):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Remove se já existir resposta para esse participante (permite sobrescrever)
    cursor.execute("DELETE FROM pos_vr_questionario WHERE participante_id = ?", (participante_id,))
    
    cursor.execute("""
        INSERT INTO pos_vr_questionario (
            participante_id, ansiedade, aversao_aranha, enjoo, realismo
        ) VALUES (?, ?, ?, ?, ?)
    """, (
        participante_id,
        dados["ansiedade"],
        dados["aversao_aranha"],
        dados["enjoo"],
        dados["realismo"]
    ))
    conn.commit()
    conn.close()

def pos_vr_questionario():
    render_back_to_home_button()
    st.title("Questionário Pós-VR (Autoavaliação)")

    participantes = buscar_participantes()
    if not participantes:
        st.warning("Nenhum participante cadastrado.")
        return

    nome_to_id = {nome: id_ for id_, nome in participantes}
    nome_selecionado = st.selectbox("Selecione o participante", list(nome_to_id.keys()))
    participante_id = nome_to_id[nome_selecionado]

    st.markdown("### Avaliações subjetivas (0 a 10)")

    with st.form("form_pos_vr"):
        ansiedade = st.slider("Ansiedade durante a experiência", 0, 10, 5)
        aversao_aranha = st.slider("Aversão a aranhas", 0, 10, 5)
        enjoo = st.slider("Náusea/Enjoo", 0, 10, 0)
        realismo = st.slider("Grau de realismo percebido", 0, 10, 5)

        if st.form_submit_button("Salvar respostas"):
            dados = {
                "ansiedade": ansiedade,
                "aversao_aranha": aversao_aranha,
                "enjoo": enjoo,
                "realismo": realismo
            }
            salvar_questionario(participante_id, dados)
            st.success("Questionário pós-VR salvo com sucesso!")

if __name__ == "__main__":
    pos_vr_questionario()

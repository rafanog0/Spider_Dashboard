import streamlit as st
import sqlite3
from datetime import datetime
from func.back_to_home import render_back_to_home_button

DB_PATH = "psicologia.db"

def inserir_participante(dados):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO participantes (
            nome, email, telefone, idade, genero, grupo,
            usa_medicacao, medicacao,
            problema_cardiaco, tontura_nausea_migranea, disponibilidade,
            pontuacao_spq, data_cadastro
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        dados["nome"], dados["email"], dados["telefone"],
        dados["idade"], dados["genero"], dados["grupo"],
        dados["usa_medicacao"], dados["medicacao"],
        dados["problema_cardiaco"], dados["tontura"],
        dados["disponibilidade"], dados["pontuacao_spq"],
        datetime.now()
    ))
    conn.commit()
    conn.close()

def cadastro_participante():
    render_back_to_home_button()
    st.title("Cadastro de Participante do Estudo")

    st.markdown("Preencha as informações abaixo conforme o protocolo do estudo.")

    with st.form("form_participante"):
        col1, col2 = st.columns(2)
        with col1:
            nome = st.text_input("Nome Completo")
            email = st.text_input("Email")
            telefone = st.text_input("Telefone")
            idade = st.number_input("Idade", min_value=14, max_value=100, step=1)
            genero = st.selectbox("Gênero", ["Feminino", "Masculino", "Outro", "Prefere não dizer"])
            grupo = st.radio("Grupo de Participação", ["fobico", "controle"])

        with col2:
            usa_medicacao = st.checkbox("Usa medicação?")
            medicacao = ""
            medicacao = st.text_input("Qual medicação utiliza?")
            problema_cardiaco = st.checkbox("Possui problema cardíaco?")
            tontura = st.checkbox("Já sentiu tontura, enjoo ou enxaqueca ao usar VR?")
            disponibilidade = st.text_area("Disponibilidade para agendamento")

            pontuacao_spq = st.slider(
                "Pontuação SPQ (Spider Phobia Questionnaire)", 0, 31, 0,
                help="Insira a pontuação total (máx: 31) conforme o questionário respondido."
            )

        submitted = st.form_submit_button("Cadastrar Participante")

        if submitted:
            if not nome or not email or not telefone:
                st.error("Preencha todos os campos obrigatórios.")
            else:
                dados = {
                    "nome": nome,
                    "email": email,
                    "telefone": telefone,
                    "idade": idade,
                    "genero": genero,
                    "grupo": grupo,
                    "usa_medicacao": usa_medicacao,
                    "medicacao": medicacao if usa_medicacao else "",
                    "problema_cardiaco": problema_cardiaco,
                    "tontura": tontura,
                    "disponibilidade": disponibilidade,
                    "pontuacao_spq": pontuacao_spq
                }
                inserir_participante(dados)
                st.success("Participante cadastrado com sucesso!")

if __name__ == "__main__":
    cadastro_participante()

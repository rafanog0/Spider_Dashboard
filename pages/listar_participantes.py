import streamlit as st
import sqlite3
import pandas as pd
from func.back_to_home import render_back_to_home_button

DB_PATH = "psicologia.db"

def buscar_participantes():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, nome, email, telefone, idade, genero, grupo,
               usa_medicacao, medicacao, problema_cardiaco,
               tontura_nausea_migranea, disponibilidade, pontuacao_spq, data_cadastro
        FROM participantes
        ORDER BY nome ASC
    """)
    colunas = [desc[0] for desc in cursor.description]
    dados = [dict(zip(colunas, linha)) for linha in cursor.fetchall()]
    conn.close()
    return dados

def atualizar_participante(part):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE participantes SET
            nome = ?, email = ?, telefone = ?, idade = ?, genero = ?, grupo = ?,
            usa_medicacao = ?, medicacao = ?, problema_cardiaco = ?,
            tontura_nausea_migranea = ?, disponibilidade = ?, pontuacao_spq = ?
        WHERE id = ?
    """, (
        part["nome"], part["email"], part["telefone"], part["idade"], part["genero"], part["grupo"],
        part["usa_medicacao"], part["medicacao"], part["problema_cardiaco"],
        part["tontura_nausea_migranea"], part["disponibilidade"], part["pontuacao_spq"], part["id"]
    ))
    conn.commit()
    conn.close()

def exibir_participantes(participantes, titulo):
    st.subheader(titulo)

    if not participantes:
        st.info("Nenhum participante encontrado.")
        return

    for p in participantes:
        with st.container():
            st.markdown(f"""
            <div style="
                border: 1px solid #444; 
                border-radius: 8px; 
                padding: 15px; 
                margin-bottom: 10px; 
                background-color: #333; 
                color: #EEE;
            ">
                <h4 style="color: #FFF; margin-bottom: 5px;">{p['nome']} ({p['genero']}, {p['idade']} anos)</h4>
                <p><strong>Email:</strong> {p['email']}</p>
                <p><strong>Telefone:</strong> {p['telefone']}</p>
                <p><strong>Grupo:</strong> <span style="color: #FF6347;">{p['grupo'].capitalize()}</span></p>
                <p><strong>Pontuação SPQ:</strong> {p['pontuacao_spq']}</p>
                <p><strong>Medicação:</strong> {"Sim – " + p['medicacao'] if p['usa_medicacao'] else "Não"}</p>
                <p><strong>Problema cardíaco:</strong> {"Sim" if p['problema_cardiaco'] else "Não"}</p>
                <p><strong>Tontura/Enjoo:</strong> {"Sim" if p['tontura_nausea_migranea'] else "Não"}</p>
                <p><strong>Disponibilidade:</strong> {p['disponibilidade']}</p>
                <p style="font-size: 12px; color: #AAA;"><em>Cadastrado em: {p['data_cadastro']}</em></p>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("✏️ Editar Informações"):
                with st.form(f"form_editar_{p['id']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        p["nome"] = st.text_input("Nome", value=p["nome"])
                        p["email"] = st.text_input("Email", value=p["email"])
                        p["telefone"] = st.text_input("Telefone", value=p["telefone"])
                        p["idade"] = st.number_input("Idade", min_value=10, max_value=100, value=p["idade"])
                        p["genero"] = st.selectbox("Gênero", ["Feminino", "Masculino", "Outro", "Prefere não dizer"], index=["Feminino", "Masculino", "Outro", "Prefere não dizer"].index(p["genero"]))
                    with col2:
                        p["grupo"] = st.radio("Grupo", ["fobico", "controle"], index=["fobico", "controle"].index(p["grupo"]))
                        p["usa_medicacao"] = st.checkbox("Usa medicação?", value=bool(p["usa_medicacao"]))
                        p["medicacao"] = st.text_input("Qual medicação?", value=p["medicacao"] if p["usa_medicacao"] else "")
                        p["problema_cardiaco"] = st.checkbox("Problema cardíaco?", value=bool(p["problema_cardiaco"]))
                        p["tontura_nausea_migranea"] = st.checkbox("Tontura/Enjoo com VR?", value=bool(p["tontura_nausea_migranea"]))

                    p["disponibilidade"] = st.text_area("Disponibilidade", value=p["disponibilidade"])
                    p["pontuacao_spq"] = st.slider("Pontuação SPQ", 0, 31, p["pontuacao_spq"])

                    if st.form_submit_button("Salvar Alterações"):
                        atualizar_participante(p)
                        st.success("Informações atualizadas com sucesso!")
                        st.rerun()

def main():
    render_back_to_home_button()
    st.title("Lista de Participantes")

    participantes = buscar_participantes()

    if participantes:
        fobicos = [p for p in participantes if p["grupo"] == "fobico"]
        controle = [p for p in participantes if p["grupo"] == "controle"]

        df = pd.DataFrame(participantes)
        st.download_button(
            label="📥 Baixar todos os dados (CSV)",
            data=df.to_csv(index=False),
            file_name="participantes.csv",
            mime="text/csv"
        )

        st.markdown("### 🔍 Filtros")
        col1, col2 = st.columns(2)
        with col1:
            nome_filtro = st.text_input("Filtrar por nome")
        with col2:
            grupo_filtro = st.selectbox("Filtrar por grupo", ["Todos", "fobico", "controle"])

        if nome_filtro:
            fobicos = [p for p in fobicos if nome_filtro.lower() in p["nome"].lower()]
            controle = [p for p in controle if nome_filtro.lower() in p["nome"].lower()]

        if grupo_filtro != "Todos":
            if grupo_filtro == "fobico":
                controle = []
            else:
                fobicos = []

        exibir_participantes(fobicos, "Grupo Fóbico")
        exibir_participantes(controle, "Grupo Controle")
    else:
        st.warning("Nenhum participante encontrado.")

if __name__ == "__main__":
    main()

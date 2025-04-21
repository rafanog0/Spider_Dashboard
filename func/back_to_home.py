import streamlit as st

# Função para renderizar o botão de voltar à página inicial
def render_back_to_home_button():
    st.sidebar.markdown(
        """
        <style>
            .back-home-btn {
                background-color: #4CA1AF;
                color: white;
                font-size: 16px;
                padding: 10px 20px;
                text-align: center;
                border-radius: 5px;
                text-decoration: none;
                display: inline-block;
                transition: all 0.3s ease-in-out;
            }
            .back-home-btn:hover {
                background-color: #3E8E92;
                transform: scale(1.05);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if st.sidebar.button("🏠 Voltar para a Página Inicial"):
        st.query_params = {"page": "home"}
        st.rerun()
import os
import streamlit as st

# Função para obter o caminho das imagens
def get_image_path(image_name):
    current_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_path, "images", image_name)

# Configuração da página
st.set_page_config(
    page_title="Painel de Psicologia",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilo customizado
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(to bottom right, #2C3E50, #4CA1AF);
            color: white;
        }
        .stButton button {
            background-color: #4CA1AF;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
            transition: all 0.3s ease-in-out;
        }
        .stButton button:hover {
            background-color: #3E8E92;
            transform: scale(1.05);
        }
        .full-width-image img {
            display: block;
            width: 100%;
            height: auto;
            margin: 20px auto;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        h1, h2, h3 {
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Imagens
brain_image_path = get_image_path("mente1.jpg")
logo_image_path = get_image_path("idp_logo.png")

# Conteúdo principal
st.title("🧠 Dashboard Integrado de Psicologia Aplicada")

if os.path.exists(brain_image_path):
    st.image(brain_image_path, use_container_width=True)
else:
    st.warning("Imagem ilustrativa não encontrada.")

st.markdown(
    """
    Este painel foi desenvolvido para auxiliar profissionais da psicologia no acompanhamento terapêutico com o uso de tecnologias modernas como:

    - **Realidade Virtual (VR)** para exposições controladas a estímulos.
    - **Sensores biométricos**, como o CardioEmotion, para coleta em tempo real de batimentos cardíacos.
    - **Scripts personalizados** para registro de eventos comportamentais em ambiente 3D.
    - **Dashboard interativo** para análise dos dados coletados ao longo das sessões.
    """
)

# Rodapé
st.markdown(
    """
    <hr style='border: 0; border-top: 1px solid #ccc; margin: 40px 0;'>
    <p style='text-align: center; font-size: 14px; color: white;'>
        Desenvolvido por <strong>IDP-DF</strong> | Fábrica de Projetos 3 | 2024<br>
        Projeto de Realidade Virtual Aplicada à Psicologia Clínica
    </p>
    """,
    unsafe_allow_html=True,
)

# 🕷️ Psicologia Experimental em Realidade Virtual

Este projeto tem como objetivo estudar a resposta fisiológica e comportamental de indivíduos com fobia de aranhas em um ambiente de Realidade Virtual (VR). A aplicação coleta dados em tempo real de frequência cardíaca e registra eventos comportamentais sincronizados com tarefas em VR.

## 🎯 Objetivo

Investigar, por meio da exposição gradual a estímulos fóbicos em VR, como se dá a evolução emocional e fisiológica dos participantes, com destaque para:

- Diferença entre grupos controle e fóbico  
- Padrões de ansiedade em tarefas progressivas  
- Sincronização de dados fisiológicos e eventos de VR

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.12**
- **Streamlit** — visualização e interface em tempo real
- **SQLite3** — banco de dados local
- **Unity 3D + OpenXR** — simulação de ambiente VR
- **CardioEmotion** — sensor de frequência cardíaca
- **Pandas, JSON, datetime, matplotlib** — processamento e análise de dados

---

## 🗃️ Estrutura do Projeto

```bash
├── app.py                          # Interface principal do dashboard
├── psicologia_final.db            # Banco de dados local
├── pages/
│   ├── cadastro_participante.py   # Formulário de cadastro
│   ├── listar_participantes.py    # Visualização e edição
│   ├── idate_form.py              # Registro da escala IDATE
│   ├── pos_vr_questionario.py     # Avaliação subjetiva pós-VR
│   ├── tempo_real.py              # Monitoramento fisiológico em tempo real
│   └── ...                        # Outras páginas (resultados de fases, etc.)
├── func/
│   ├── salva_cortado.py           # Leitura e extração de BPM
│   ├── back_to_home.py            # Navegação para tela inicial
│   └── ...
└── README.md

```



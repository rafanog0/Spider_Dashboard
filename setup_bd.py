# setup_bd.py
# -*- coding: utf-8 -*-
"""
Script para criação do banco de dados SQLite para o projeto de pesquisa VR.
Este schema foi ajustado para ser compatível com o script de análise (analise.py),
principalmente na tabela 'idate' para permitir a diferenciação entre as
aplicações pré e pós-experimento.
"""

import sqlite3

# Caminho onde o banco de dados será criado ou acessado.
db_path = "psicologia.db"

# Tenta estabelecer conexão com o arquivo do banco de dados.
try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # --- Bloco de Criação das Tabelas ---
    # O executescript permite rodar múltiplos comandos SQL de uma vez.
    cursor.executescript("""
    
    -- Tabela de Participantes: armazena dados demográficos e de triagem.
    CREATE TABLE IF NOT EXISTS participantes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nome TEXT,
        email TEXT UNIQUE,
        telefone TEXT,
        idade INTEGER,
        genero TEXT,
        grupo TEXT, -- Esperado: 'fobico' ou 'controle' para compatibilidade com a análise.
        usa_medicacao BOOLEAN,
        medicacao TEXT,
        problema_cardiaco BOOLEAN,
        tontura_nausea_migranea BOOLEAN,
        disponibilidade TEXT,
        pontuacao_spq INTEGER,
        data_cadastro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Tabela IDATE: armazena respostas dos questionários de ansiedade.
    -- AJUSTE CRÍTICO: A coluna 'tipo' foi alterada para TEXT sem restrição CHECK,
    -- permitindo os valores necessários para a análise: 
    -- 'idate_estado_pre', 'idate_estado_pos', 'idate_traco'.
    CREATE TABLE IF NOT EXISTS idate (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        participante_id INTEGER NOT NULL,
        tipo TEXT NOT NULL,
        questao INTEGER NOT NULL,
        resposta INTEGER CHECK(resposta BETWEEN 1 AND 4),
        FOREIGN KEY (participante_id) REFERENCES participantes(id) ON DELETE CASCADE
    );

    -- Tabela Pós-VR: armazena a autoavaliação subjetiva após o experimento.
    CREATE TABLE IF NOT EXISTS pos_vr_questionario (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        participante_id INTEGER NOT NULL,
        ansiedade INTEGER CHECK(ansiedade BETWEEN 0 AND 10),
        aversao_aranha INTEGER CHECK(aversao_aranha BETWEEN 0 AND 10),
        enjoo INTEGER CHECK(enjoo BETWEEN 0 AND 10),
        realismo INTEGER CHECK(realismo BETWEEN 0 AND 10),
        FOREIGN KEY (participante_id) REFERENCES participantes(id) ON DELETE CASCADE
    );

    -- Tabela Fase 1: dados da primeira fase de exposição.
    CREATE TABLE IF NOT EXISTS fase1 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        participante_id INTEGER NOT NULL,
        session_bpm TEXT, -- Armazena a série de BPM, geralmente como JSON ou CSV.
        tempo_finalizacao REAL,
        tempo_encontrar_aranha REAL,
        FOREIGN KEY (participante_id) REFERENCES participantes(id) ON DELETE CASCADE
    );

    -- Tabela Fase 2: dados da segunda fase de exposição.
    CREATE TABLE IF NOT EXISTS fase2 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        participante_id INTEGER NOT NULL,
        session_bpm TEXT,
        tempo_finalizacao REAL,
        tempo_encontrar_aranha REAL,
        FOREIGN KEY (participante_id) REFERENCES participantes(id) ON DELETE CASCADE
    );

    -- Tabela Fase 3: dados da terceira e mais complexa fase.
    CREATE TABLE IF NOT EXISTS fase3 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        participante_id INTEGER NOT NULL,
        session_bpm TEXT,
        tempo_finalizacao REAL,
        tempo_encontrar_aranha_1 REAL,
        tempo_encontrar_aranha_2 REAL,
        tempo_encontrar_aranha_3 REAL,
        tempo_encontrar_aranha_4 REAL,
        tempo_encontrar_aranha_5 REAL,
        tempo_encontrar_aranha_6 REAL,
        tempo_encontrar_aranha_7 REAL,
        tempo_encontrar_aranha_8 REAL,
        tempo_encontrar_aranha_9 REAL,
        tempo_encontrar_aranha_10 REAL,
        tempo_encontrar_aranha_11 REAL,
        tempo_encontrar_aranha_12 REAL,
        tempo_encontrar_aranha_13 REAL,
        tempo_encontrar_aranha_14 REAL,
        tempo_encontrar_aranha_15 REAL,
        tempo_encontrar_5a_aranha REAL,
        total_aranhas_encontradas INTEGER,
        FOREIGN KEY (participante_id) REFERENCES participantes(id) ON DELETE CASCADE
    );
    """)

    # Salva (commit) todas as alterações feitas no banco de dados.
    conn.commit()
    print(f"Banco de dados '{db_path}' verificado/criado com sucesso.")

except sqlite3.Error as e:
    print(f"Ocorreu um erro ao trabalhar com o banco de dados: {e}")

finally:
    # Garante que a conexão seja sempre fechada, mesmo se ocorrer um erro.
    if conn:
        conn.close()
        print("Conexão com o SQLite foi fechada.")


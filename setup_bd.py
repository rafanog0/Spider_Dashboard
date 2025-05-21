import sqlite3

# Caminho onde o banco será salvo (pode alterar o caminho se quiser)
db_path = "psicologia.db"

# Conexão com o banco
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Criação das tabelas
cursor.executescript("""
-- Participantes
CREATE TABLE IF NOT EXISTS participantes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nome TEXT,
    email TEXT,
    telefone TEXT,
    idade INTEGER,
    genero TEXT,
    grupo TEXT,
    usa_medicacao BOOLEAN,
    medicacao TEXT,
    problema_cardiaco BOOLEAN,
    tontura_nausea_migranea BOOLEAN,
    disponibilidade TEXT,
    pontuacao_spq INTEGER,
    data_cadastro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- IDATE (Estado e Traço)
CREATE TABLE IF NOT EXISTS idate (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    participante_id INTEGER,
    tipo TEXT CHECK(tipo IN ('estado', 'traço')),
    questao INTEGER,
    resposta INTEGER CHECK(resposta BETWEEN 1 AND 4),
    FOREIGN KEY (participante_id) REFERENCES participantes(id)
);

-- Pós-VR (autoavaliação)
CREATE TABLE IF NOT EXISTS pos_vr_questionario (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    participante_id INTEGER,
    ansiedade INTEGER CHECK(ansiedade BETWEEN 0 AND 10),
    aversao_aranha INTEGER CHECK(aversao_aranha BETWEEN 0 AND 10),
    enjoo INTEGER CHECK(enjoo BETWEEN 0 AND 10),
    realismo INTEGER CHECK(realismo BETWEEN 0 AND 10),
    FOREIGN KEY (participante_id) REFERENCES participantes(id)
);

-- Fase 1
CREATE TABLE IF NOT EXISTS fase1 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    participante_id INTEGER,
    session_bpm TEXT,
    tempo_finalizacao REAL,
    tempo_encontrar_aranha REAL,
    FOREIGN KEY (participante_id) REFERENCES participantes(id)
);

-- Fase 2
CREATE TABLE IF NOT EXISTS fase2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    participante_id INTEGER,
    session_bpm TEXT,
    tempo_finalizacao REAL,
    tempo_encontrar_aranha REAL,
    FOREIGN KEY (participante_id) REFERENCES participantes(id)
);

-- Fase 3
CREATE TABLE IF NOT EXISTS fase3 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    participante_id INTEGER,
    session_bpm TEXT,
    tempo_finalizacao REAL,
    tempo_encontrar_aranha_1 TIMESTAMP,
    tempo_encontrar_aranha_2 TIMESTAMP,
    tempo_encontrar_aranha_3 TIMESTAMP,
    tempo_encontrar_aranha_4 TIMESTAMP,
    tempo_encontrar_aranha_5 TIMESTAMP,
    tempo_encontrar_aranha_6 TIMESTAMP,
    tempo_encontrar_aranha_7 TIMESTAMP,
    tempo_encontrar_aranha_8 TIMESTAMP,
    tempo_encontrar_aranha_9 TIMESTAMP,
    tempo_encontrar_aranha_10 TIMESTAMP,
    tempo_encontrar_aranha_11 TIMESTAMP,
    tempo_encontrar_aranha_12 TIMESTAMP,
    tempo_encontrar_aranha_13 TIMESTAMP,
    tempo_encontrar_aranha_14 TIMESTAMP,
    tempo_encontrar_aranha_15 TIMESTAMP,
    tempo_encontrar_5a_aranha REAL,
    total_aranhas_encontradas INTEGER,
    FOREIGN KEY (participante_id) REFERENCES participantes(id)
);
""")

# Finaliza
conn.commit()
conn.close()

print(f"Banco de dados criado com sucesso: {db_path}")

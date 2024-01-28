CREATE TABLE IF NOT EXISTS freelance_jobs (
    id SERIAL PRIMARY KEY,
    horas_esperadas INTEGER NOT NULL,
    preco INTEGER NOT NULL,
    finalizado INTEGER NOT NULL
);

INSERT INTO freelance_jobs (horas_esperadas, preco, finalizado) VALUES (100, 1000, 0);

CREATE TABLE IF NOT EXISTS marketing_investimentos (
    id SERIAL PRIMARY KEY,
    receita_cliente REAL,
    anuidade_emprestimo REAL,
    anos_casa_propria REAL,
    telefone_trab INTEGER,
    avaliacao_cidade REAL,
    score_1 REAL,
    score_2 REAL,
    score_3 REAL,
    score_social REAL,
    troca_telefone REAL,
    inadimplente INTEGER NOT NULL
);

INSERT INTO marketing_investimentos VALUES (
    1,
    0.0,
    0.0,
    0.0,
    0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0,
    0
);

CREATE TABLE IF NOT EXISTS vendas_carros (
    id SERIAL PRIMARY KEY,
    milhas_por_ano INTEGER NOT NULL,
    ano_do_modelo INTEGER NOT NULL,
    preco REAL,
    vendido INTEGER NOT NULL
);


INSERT INTO vendas_carros VALUES (
    1,
    10000,
    2000,
    40000.0,
    1
);
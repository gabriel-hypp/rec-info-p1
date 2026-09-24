"""
Lê docs e consultas de data/raw/ e gera versões pré-processadas
em data/processed/, de acordo com o exigido no enunciado usando
a biblioteca NLTK:

    1. nada       -> apenas tokenização + lowercase
    2. stopwords  -> só remoção de stopwords
    3. stemming   -> só stemming
    4. ambas      -> stopwords + stemming

Uso: 
    python -m src.preprocessamento
"""

import os
import json
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "dados", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "dados", "processed")

CONFIGS = ["nada", "stopwords", "stemming", "ambas"]

def nltk_resources():
    """
    Verifica presença dos recursos necessários para o NLTK. 
    Necessário na primeira execução.
    """
    for resource in ["punkt", "punkt_tab", "stopwords"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def preprocess_text(text, remove_stopwords=False, apply_stemming=False, stop_words=None, stemmer=None):
    """
    Aplica tokenização + lowercase sempre; stopwords e stemming são
    opcionais conforme a configuração escolhida.
    """
    tokens = word_tokenize(text.lower())
    # Remove tokens que não são palavras (pontuação, números)
    tokens = [t for t in tokens if t.isalpha()]

    if remove_stopwords:
        tokens = [t for t in tokens if t not in stop_words]

    if apply_stemming:
        tokens = [stemmer.stem(t) for t in tokens]

    return tokens

def config_params(config_name):
    """Retorna um dicionário com os parâmetros de acordo com a config."""
    return {
        "nada": dict(remove_stopwords=False, apply_stemming=False),
        "stopwords": dict(remove_stopwords=True, apply_stemming=False),
        "stemming": dict(remove_stopwords=False, apply_stemming=True),
        "ambas": dict(remove_stopwords=True, apply_stemming=True),
    }[config_name]

def process_dataframe(df, config_name, stop_words, stemmer):
    """
    Processa os textos de docs e consultas, de acordo com a configuração,
    e retorna uma lista de tokens por linha.
    """
    params = config_params(config_name)
    results = []
    for _, row in df.iterrows():
        full_text = str(row["text"])
        tokens = preprocess_text(full_text, stop_words=stop_words, stemmer=stemmer, **params)
        results.append(tokens)
    return results

def main():
    # Verifica presença dos recursos necessários do NLTK e baixa se necessário  
    nltk_resources()
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Carrega stopwords e stemmer
    stop_words = set(stopwords.words("english"))
    stemmer = SnowballStemmer("english")

    # Carrega datasets
    docs = pd.read_csv(os.path.join(RAW_DIR, "docs.csv"))
    queries = pd.read_csv(os.path.join(RAW_DIR, "queries.csv"))

    for config in CONFIGS:
        print(f"Processando configuração: {config}")

        doc_tokens = process_dataframe(docs, config, stop_words, stemmer)
        query_tokens = process_dataframe(queries, config, stop_words, stemmer)

        # Salva resultados em JSON, com doc_id e query_id como chaves
        docs_out = {str(doc_id): toks for doc_id, toks in zip(docs["doc_id"], doc_tokens)}
        queries_out = {str(q_id): toks for q_id, toks in zip(queries["query_id"], query_tokens)}

        with open(os.path.join(PROCESSED_DIR, f"docs_{config}.json"), "w") as f:
            json.dump(docs_out, f)
        with open(os.path.join(PROCESSED_DIR, f"queries_{config}.json"), "w") as f:
            json.dump(queries_out, f)

        print(f"-> salvos docs_{config}.json e queries_{config}.json")

if __name__ == "__main__":
    main()
"""
Compara as 4 configurações de pré-processamento definidas em
src/preprocessing.py:
    - nada       (sem stopwords, sem stemming)
    - stopwords  (com remoção de stopwords)
    - stemming   (com stemming)
    - ambas      (stopwords + stemming)

Modelo fixo vetorial para as 4 configurações, a com maior MAP será 
usada como padrão nos testes para os outros requisitos do trabalho.

Uso:
    python -m experiments.run_preprocessing_comparison
"""

import os
import json
import pandas as pd

from src.modelo_vetorial import VectorModel
from src.modelo_probabilistico import BM25Model
from src.metrics import relevant_sets, avaliacao

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DIR = os.path.join(BASE_DIR, "dados", "processed")
RAW_DIR = os.path.join(BASE_DIR, "dados", "raw")
RESULTS_DIR = os.path.join(BASE_DIR, "resultados")

CONFIGS = ["nada", "stopwords", "stemming", "ambas"]

def load_processed(config_name):
    with open(os.path.join(PROCESSED_DIR, f"docs_{config_name}.json")) as f:
        docs = json.load(f)
    with open(os.path.join(PROCESSED_DIR, f"queries_{config_name}.json")) as f:
        queries = json.load(f)
    return docs, queries

def run(relevant_set, k=10):
    """
    Roda o modelo indicado para as 4 configurações e retorna:
        - por_query_by_config: {config_name: Dataframe por consulta}
        - aggregated_df: Dataframe com uma linha por configuração
    """
    por_query_by_config = {}
    agg_rows = []

    for config_name in CONFIGS:
        print(f"Processando configuração: {config_name}")
        docs, queries = load_processed(config_name)

        model = VectorModel().fit(docs)
        rankings = model.rank_all_queries(queries, top_k=None)
        por_query, aggregated = avaliacao(rankings, relevant_set, k=k)

        por_query_df = pd.DataFrame(por_query)
        por_query_by_config[config_name] = por_query_df

        agg_rows.append({"config": config_name, **aggregated})

    aggregated_df = pd.DataFrame(agg_rows)
    return por_query_by_config, aggregated_df

def save_results(por_query_by_config, aggregated_df):
    por_query_dir = os.path.join(RESULTS_DIR, "por_query")
    aggregated_dir = os.path.join(RESULTS_DIR, "agregados")
    os.makedirs(por_query_dir, exist_ok=True)
    os.makedirs(aggregated_dir, exist_ok=True)

    for config_name, df in por_query_by_config.items():
        out_path = os.path.join(
            por_query_dir, f"preprocessing_{config_name}.csv"
        )
        df.to_csv(out_path, index=False)

    agg_path = os.path.join(aggregated_dir, f"compara_preprocessamento.csv")
    aggregated_df.to_csv(agg_path, index=False)
    print(f"\n[Resultados agregados salvos em {agg_path}")
    print(aggregated_df)


def main():
    qrels_df = pd.read_csv(os.path.join(RAW_DIR, "qrels.csv"))
    relevant_set = relevant_sets(qrels_df)

    vector_por_query, vector_agg = run(relevant_set, k=50)
    save_results(vector_por_query, vector_agg)

    best_config = vector_agg.sort_values("MAP", ascending=False).iloc[0]["config"]
    print(f"\nConfiguração com maior MAP (Vetorial): {best_config}")

if __name__ == "__main__":
    main()
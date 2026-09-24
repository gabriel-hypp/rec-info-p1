"""
Comparação quantitativa entre Modelo Vetorial e BM25.

Script para carregar dados processados -> rodar modelos -> avaliar -> 
salvar resultados por consulta e agregados em resultados/.

Uso:
    python -m testes.compara_modelos
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
 
# Configuração de pré-processamento escolhida como "padrão"
# possibilidade trocar entre: "nada", "stopwords", "stemming", "ambas"
CONFIG_NAME = "ambas"

def load_processed(config_name):
    with open(os.path.join(PROCESSED_DIR, f"docs_{config_name}.json")) as f:
        docs = json.load(f)
    with open(os.path.join(PROCESSED_DIR, f"queries_{config_name}.json")) as f:
        queries = json.load(f)
    return docs, queries
 
def main():
    docs, queries = load_processed(CONFIG_NAME)
    qrels_df = pd.read_csv(os.path.join(RAW_DIR, "qrels.csv"))
    # Carrega conjunto de documentos relevantes por consulta
    relevant_set = relevant_sets(qrels_df)
 
    # --- Modelo Vetorial ---
    vector_model = VectorModel().fit(docs)
    vector_rankings = vector_model.rank_all_queries(queries, top_k=None)
    vector_por_query, vector_agg = avaliacao(vector_rankings, relevant_set, k=10)
 
    # --- BM25 ---
    bm25_model = BM25Model(k1=1.2, b=0.75).fit(docs)
    bm25_rankings = bm25_model.rank_all_queries(queries, top_k=None)
    bm25_por_query, bm25_agg = avaliacao(bm25_rankings, relevant_set, k=10)
 
    # --- Salvar resultados por consulta ---
    os.makedirs(os.path.join(RESULTS_DIR, "por_query"), exist_ok=True)
    pd.DataFrame(vector_por_query).to_csv(
        os.path.join(RESULTS_DIR, "por_query", "modelo_vetorial.csv"), index=False)
    pd.DataFrame(bm25_por_query).to_csv(
        os.path.join(RESULTS_DIR, "por_query", "bm25.csv"), index=False)
 
    # --- Salvar resultados agregados ---
    os.makedirs(os.path.join(RESULTS_DIR, "agregados"), exist_ok=True)
    agg_df = pd.DataFrame([
        {"model": "Vector", **vector_agg},
        {"model": "BM25", **bm25_agg},
    ])
    agg_df.to_csv(
        os.path.join(RESULTS_DIR, "agregados", f"compara_modelos_{CONFIG_NAME}.csv"),
        index=False,
    )
    
    # Resultados agregados
    print(agg_df)
 
    # --- Diferenças por consulta ---

    # Cria dataframe com diferença de AP por consulta entre BM25 e Vetorial
    diff_df = pd.DataFrame({
        "query_id": [r["query_id"] for r in vector_por_query],
        "AP_vector": [r["AP"] for r in vector_por_query],
    }).merge(
        pd.DataFrame({
            "query_id": [r["query_id"] for r in bm25_por_query],
            "AP_bm25": [r["AP"] for r in bm25_por_query],
        }),
        on="query_id"
    )
    diff_df["AP_diff (bm25 - vector)"] = diff_df["AP_bm25"] - diff_df["AP_vector"]
    diff_df["abs_diff"] = diff_df["AP_diff (bm25 - vector)"].abs()
    diff_df = diff_df.sort_values("abs_diff", ascending=False)
    diff_df.to_csv(
        os.path.join(RESULTS_DIR, "agregados", f"ap_diff_por_query_{CONFIG_NAME}.csv"),
        index=False,
    )
 
    print(f'\n{diff_df.head(5)})')
 
if __name__ == "__main__":
    main()
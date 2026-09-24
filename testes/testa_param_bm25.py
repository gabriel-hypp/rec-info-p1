"""
Variação dos parâmetros do BM25.

Compara as configurações por MAP e por Precision@10. Também localiza, para 
um k1 fixo, a consulta cujo Top-10 mais muda quando b varia de 0 para 1

Uso:
    python -m testes.testa_param_bm25
"""

import os
import json
import itertools
import pandas as pd

from src.modelo_probabilistico import BM25Model
from src.metrics import relevant_sets, avaliacao

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DIR = os.path.join(BASE_DIR, "dados", "processed")
RAW_DIR = os.path.join(BASE_DIR, "dados", "raw")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

CONFIG_NAME = "ambas"

K1_VALUES = [0.5, 1.2, 2.0]
B_VALUES = [0.0, 0.75, 1.0]

# k1 fixo pra testar b
K1_FIXED = 2.0 # 2 tem relevante, 1.2 muda mais

def load_processed(config_name):
    with open(os.path.join(PROCESSED_DIR, f"docs_{config_name}.json")) as f:
        docs = json.load(f)
    with open(os.path.join(PROCESSED_DIR, f"queries_{config_name}.json")) as f:
        queries = json.load(f)
    return docs, queries

def run_grid(docs, queries, relevant_set, k=10):
    """
    Roda k1 x b, ajusta só uma vez e calcula o MAP e Precision@10 para o Top-10
    """
    model = BM25Model().fit(docs)
    rows = []
    rankings_cache = {}

    for k1, b in itertools.product(K1_VALUES, B_VALUES):
        model.set_params(k1=k1, b=b)
        rankings = model.rank_all_queries(queries, top_k=None)
        rankings_cache[(k1, b)] = rankings

        _, aggregated = avaliacao(rankings, relevant_set, k=k)
        rows.append({"k1": k1, "b": b, **aggregated})

    grid_df = pd.DataFrame(rows)
    return grid_df, rankings_cache

def most_affected(rankings_cache, k1=K1_FIXED, k=10):
    """
    Para k1 fixo, compara o Top-10 de cada consulta em b=0 vs. b=1 usando a
    sobreposição (Jaccard) dos conjuntos de doc_ids. Retorna a consulta com
    menor sobreposição, onde o ranking mais mudou.
    """
    ranking_b0 = rankings_cache[(k1, 0.0)]
    ranking_b1 = rankings_cache[(k1, 1.0)]

    rows = []
    for query_id in ranking_b0.keys():
        docs_b0 = {doc_id for doc_id, _ in ranking_b0[query_id][:k]}
        docs_b1 = {doc_id for doc_id, _ in ranking_b1[query_id][:k]}
        union = docs_b0 | docs_b1
        jaccard = len(docs_b0 & docs_b1) / len(union) if union else 1.0
        rows.append({"query_id": query_id, "jaccard_b0_vs_b1": round(jaccard, 3)})

    overlap_df = pd.DataFrame(rows).sort_values("jaccard_b0_vs_b1")
    most_affected_query = overlap_df.iloc[0]["query_id"]
    return most_affected_query, overlap_df


def show_ranking_comparison(query_id, rankings_cache, relevant_set, k1=K1_FIXED, k=10):
    """Mostra o Top-10 lado a lado para b=0, b=0.75 e b=1, marcando relevância."""
    rel_set = relevant_set.get(str(query_id), set())
    print(f"\nComparação de ranking para query_id={query_id} (k1={k1}), variando b:")
    for b in B_VALUES:
        ranking = rankings_cache[(k1, b)][query_id][:k]
        doc_ids = [f"{doc_id}{'*' if doc_id in rel_set else ''}" for doc_id, _ in ranking]
        print(f"  b={b}: {doc_ids}")
    print("  (* = documento relevante segundo os qrels)")


def main():
    docs, queries = load_processed(CONFIG_NAME)
    qrels_df = pd.read_csv(os.path.join(RAW_DIR, "qrels.csv"))
    relevant_set = relevant_sets(qrels_df)

    grid_df, rankings_cache = run_grid(docs, queries, relevant_set, k=10)

    os.makedirs(os.path.join(RESULTS_DIR, "agregados"), exist_ok=True)
    grid_path = os.path.join(RESULTS_DIR, "agregados", f"teste_param_b25_{CONFIG_NAME}.csv")
    grid_df.to_csv(grid_path, index=False)

    print("Resultados da grade k1 x b:")
    print(grid_df.to_string(index=False))
    print(f"\nSalvo em {grid_path}")

    best_row = grid_df.sort_values("MAP", ascending=False).iloc[0]
    print(f"\nMelhor combinação por MAP: k1={best_row['k1']}, b={best_row['b']} "
          f"(MAP={best_row['MAP']:.4f}, precision@10={best_row[f'precision@10']:.4f})")

    # tabelas pivotadas uteis para visualização
    pivot_map = grid_df.pivot(index="k1", columns="b", values="MAP")
    pivot_p10 = grid_df.pivot(index="k1", columns="b", values=f"precision@10")
    print("\nMAP por (k1, b):")
    print(pivot_map)
    print(f"\nPrecision@10 por (k1, b):")
    print(pivot_p10)

    # consulta mais afetada pela variação de b (k1 fixo)
    query_id, overlap_df = most_affected(rankings_cache, k1=K1_FIXED, k=10)
    overlap_path = os.path.join(RESULTS_DIR, "agregados", f"efeito_variacao_b_{CONFIG_NAME}.csv")
    overlap_df.to_csv(overlap_path, index=False)

    print(f"\nConsulta mais afetada pela variação de b (k1={K1_FIXED}): "
          f"query_id={query_id}")
    show_ranking_comparison(query_id, rankings_cache, relevant_set, k1=K1_FIXED, k=10)

if __name__ == "__main__":
    main()
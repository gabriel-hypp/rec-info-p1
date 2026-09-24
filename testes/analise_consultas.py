"""
Seleciona consultas para os três casos pedidos e mostra o Top 5 
de documentos retornados por cada modelo, indicando quais são relevantes.

Casos:
    i)   2 consultas em que o BM25 é superior ao Vetorial
    ii)  2 consultas em que o Vetorial é superior ao BM25
    iii) 2 consultas em que ambos têm desempenho insatisfatório

Ssa a diferença de AP (bm25 - vector) ou filtra AP abaixo de um limiar

Uso:
    python -m testes.analise_consultas
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

CONFIG_NAME = "ambas"  # "nada", "stopwords", "stemming", "ambas"

# Limiar de AP para desempenho insatisfatório
LIMIAR = 0.10

def load_processed(config_name):
    with open(os.path.join(PROCESSED_DIR, f"docs_{config_name}.json")) as f:
        docs = json.load(f)
    with open(os.path.join(PROCESSED_DIR, f"queries_{config_name}.json")) as f:
        queries = json.load(f)
    return docs, queries

def select_queries(vector_por_query, bm25_por_query):
    """
    Recebe as listas de dicts retornadas por avaliacao() para cada
    modelo e seleciona as consultas para os 3 casos do requisito 6.
    """
    # Junta os dois dfs por query_id, renomeando colunas de ap
    v_df = pd.DataFrame(vector_por_query)[["query_id", "AP"]].rename(columns={"AP": "AP_vector"})
    b_df = pd.DataFrame(bm25_por_query)[["query_id", "AP"]].rename(columns={"AP": "AP_bm25"})
    merged = v_df.merge(b_df, on="query_id")
    merged["diff"] = merged["AP_bm25"] - merged["AP_vector"]

    # Seleciona as 2 consultas com maior diferença
    bm25_better = merged.sort_values("diff", ascending=False).head(2)
    vector_better = merged.sort_values("diff", ascending=True).head(2)

    # Evita consultas repetidas em 2 casos diferentes
    already_selected = set(bm25_better["query_id"]) | set(vector_better["query_id"])
    remaining = merged[~merged["query_id"].isin(already_selected)]

    bads = remaining[
        (remaining["AP_vector"] < LIMIAR) &
        (remaining["AP_bm25"] < LIMIAR)
    ].copy()
    bads["combined"] = bads["AP_vector"] + bads["AP_bm25"]
    bads = bads.sort_values("combined").head(2)

    return {
        "bm25_better": bm25_better,
        "vector_better": vector_better,
        "bads": bads,
    }

def inspect_query(query_id, docs_raw, docs_tokens, query_tokens_dict,
                   vector_model, bm25_model, relevant_set, n=5):
    """
    Roda os dois modelos para uma consulta específica e retorna uma tabela
    com o Top 5 de cada modelo, indicando relevância e um preview do texto
    do documento.
    """
    q_tokens = query_tokens_dict[str(query_id)]

    vector_ranking = vector_model.rank(q_tokens, top_k=n)
    bm25_ranking = bm25_model.rank(q_tokens, top_k=n)

    rows = []
    for rank_pos, (doc_id, score) in enumerate(vector_ranking, start=1):
        rows.append({
            "query_id": query_id, "model": "Vector", "rank": rank_pos,
            "doc_id": doc_id, "score": score,
            "relevant": doc_id in relevant_set,
            "preview": docs_raw.get(str(doc_id), "")[:120],
        })
    for rank_pos, (doc_id, score) in enumerate(bm25_ranking, start=1):
        rows.append({
            "query_id": query_id, "model": "BM25", "rank": rank_pos,
            "doc_id": doc_id, "score": score,
            "relevant": doc_id in relevant_set,
            "preview": docs_raw.get(str(doc_id), "")[:120],
        })
    return rows


def main():
    docs, queries = load_processed(CONFIG_NAME)
    qrels_df = pd.read_csv(os.path.join(RAW_DIR, "qrels.csv"))
    relevant_set = relevant_sets(qrels_df)

    # lookup de título/texto original para preview
    docs_raw_df = pd.read_csv(os.path.join(RAW_DIR, "docs.csv"))
    docs_raw = {
        str(row["doc_id"]): str(row["title"])
        for _, row in docs_raw_df.iterrows()
    }
    queries_raw_df = pd.read_csv(os.path.join(RAW_DIR, "queries.csv"))
    queries_raw = {
        str(row["query_id"]): str(row["text"])
        for _, row in queries_raw_df.iterrows()
    }

    vector_model = VectorModel().fit(docs)
    bm25_model = BM25Model(k1=1.2, b=0.75).fit(docs)

    vector_rankings = vector_model.rank_all_queries(queries, top_k=None)
    bm25_rankings = bm25_model.rank_all_queries(queries, top_k=None)

    vector_per_query, _ = avaliacao(vector_rankings, relevant_set, k=10)
    bm25_per_query, _ = avaliacao(bm25_rankings, relevant_set, k=10)

    selected = select_queries(vector_per_query, bm25_per_query)

    all_rows = []
    for case_name, df in selected.items():
        print(f"\n=== Caso: {case_name} ===")
        for _, row in df.iterrows():
            q_id = row["query_id"]
            print(f"\nquery_id={q_id} | AP_vector={row['AP_vector']:.3f} | "
                  f"AP_bm25={row['AP_bm25']:.3f}")
            print(f"texto da consulta: {queries_raw.get(str(q_id), '')}")

            rel_set = relevant_set.get(str(q_id), set())
            rows = inspect_query(q_id, docs_raw, docs, queries,
                                  vector_model, bm25_model, rel_set)
            for r in rows:
                r["case"] = case_name
            all_rows.extend(rows)

            case_df = pd.DataFrame(rows)
            print(case_df[["model", "rank", "doc_id", "score", "relevant", "preview"]]
                  .to_string(index=False))

    out_df = pd.DataFrame(all_rows)
    out_path = os.path.join(RESULTS_DIR, "agregados", "analise_consultas.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nResultado completo salvo em {out_path}")


if __name__ == "__main__":
    main()

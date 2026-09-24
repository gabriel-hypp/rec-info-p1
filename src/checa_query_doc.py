"""
Utilitário simples para inspecionar uma consulta e um documento específicos
(dados os IDs), mostrando texto original, tokens pré-processados e as
estatísticas (TF, DF, IDF, tamanho do documento) que embasam as hipóteses
dos requisitos 5 e 6.
"""

import os
import json
import pandas as pd

from src.modelo_probabilistico import BM25Model
from src.metrics import relevant_sets

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DIR = os.path.join(BASE_DIR, "dados", "processed")
RAW_DIR = os.path.join(BASE_DIR, "dados", "raw")

# Editar ids
QUERY_ID = "81"
DOC_ID = "486"
CONFIG_NAME = "ambas"  # "nada", "stopwords", "stemming", "ambas"
K1, B = 1.2, 0.75

def _load(config_name):
    with open(os.path.join(PROCESSED_DIR, f"docs_{config_name}.json")) as f:
        docs_tokens = json.load(f)
    with open(os.path.join(PROCESSED_DIR, f"queries_{config_name}.json")) as f:
        queries_tokens = json.load(f)

    docs_raw_df = pd.read_csv(os.path.join(RAW_DIR, "docs.csv"))
    queries_raw_df = pd.read_csv(os.path.join(RAW_DIR, "queries.csv"))
    qrels_df = pd.read_csv(os.path.join(RAW_DIR, "qrels.csv"))

    docs_raw_df["doc_id"] = docs_raw_df["doc_id"].astype(str)
    queries_raw_df["query_id"] = queries_raw_df["query_id"].astype(str)

    return docs_tokens, queries_tokens, docs_raw_df, queries_raw_df, qrels_df

def inspect(query_id, doc_id, config_name=CONFIG_NAME, k1=K1, b=B):
    query_id, doc_id = str(query_id), str(doc_id)
    docs_tokens, queries_tokens, docs_raw_df, queries_raw_df, qrels_df = _load(config_name)

    bm25_model = BM25Model(k1=k1, b=b).fit(docs_tokens)
    relevant_set = relevant_sets(qrels_df)

    query_text_row = queries_raw_df[queries_raw_df["query_id"] == query_id]
    doc_row = docs_raw_df[docs_raw_df["doc_id"] == doc_id]

    query_text = query_text_row["text"].values[0] if len(query_text_row) else "(query_id não encontrado)"
    doc_title = doc_row["title"].values[0] if len(doc_row) else "(doc_id não encontrado)"
    doc_text = doc_row["text"].values[0] if len(doc_row) else ""

    query_tokens = queries_tokens.get(query_id, [])
    is_relevant = doc_id in relevant_set.get(query_id, set())

    doc_len = bm25_model.doc_tams.get(doc_id)
    avgdl = bm25_model.avgdl
    len_ratio = round(doc_len / avgdl, 2) if doc_len and avgdl else None

    freqs = bm25_model.doc_freqs.get(doc_id, {})
    term_rows = []
    for term in sorted(set(query_tokens)):
        tf = freqs.get(term, 0)
        df = bm25_model.df.get(term, 0)
        idf = bm25_model._idf(term)
        contribution = 0.0
        if tf > 0:
            denom = tf + k1 * (1 - b + b * doc_len / avgdl)
            contribution = round(idf * (tf * (k1 + 1) / denom), 3)
        term_rows.append({
            "term": term, "tf_in_doc": tf, "df_collection": df,
            "idf_bm25": round(idf, 3), "bm25_contribution": contribution,
        })

    bm25_score = round(bm25_model._score(doc_id, query_tokens), 4) if doc_id in bm25_model.doc_ids else None

    print("=" * 80)
    print(f"QUERY {query_id}: {query_text}")
    print(f"tokens processados: {query_tokens}")
    print("-" * 80)
    print(f"DOC {doc_id}: {doc_title}")
    print(f"texto (abstract): {doc_text[:500]}{'...' if len(doc_text) > 500 else ''}")
    print("-" * 80)
    print(f"Relevante para essa consulta (qrels)? {'SIM' if is_relevant else 'NÃO'}")
    print(f"Tamanho do documento: {doc_len} tokens (avgdl da coleção: {round(avgdl, 2)}, razão: {len_ratio})")
    print(f"Score BM25 total (k1={k1}, b={b}): {bm25_score}")
    print("-" * 80)
    print("Diagnóstico por termo da consulta:")
    print(pd.DataFrame(term_rows).to_string(index=False) if term_rows else "(consulta sem tokens)")
    print("=" * 80)


if __name__ == "__main__":
    inspect(QUERY_ID, DOC_ID)
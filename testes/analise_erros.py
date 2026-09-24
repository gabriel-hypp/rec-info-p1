"""
Localiza: pelo menos 2 documentos não relevantes nas primeiras posições do ranking (falsos positivos)
          pelo menos 1 documento revelante que não aparece nas primeiras (falso negativo)

Imprime, para cada caso selecionado TF/DF/IDF dos termos da consulta no documento e tamanho do
documento vs. avgdl pra ajudar a investigar a causa

Uso:
    python -m testes.analise_erros
"""

import os
import json
import pandas as pd

from src.modelo_probabilistico import BM25Model
from src.metrics import relevant_sets

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DIR = os.path.join(BASE_DIR, "dados", "processed")
RAW_DIR = os.path.join(BASE_DIR, "dados", "raw")
RESULTS_DIR = os.path.join(BASE_DIR, "resultados")

CONFIG_NAME = "ambas"
TOP_K = 10

# até que posição um documento não relevante conta como "primeiras posições"
FP_CUTOFF = 3
N_FP = 2
N_FN = 1

def load_processed(config_name):
    with open(os.path.join(PROCESSED_DIR, f"docs_{config_name}.json")) as f:
        docs = json.load(f)
    with open(os.path.join(PROCESSED_DIR, f"queries_{config_name}.json")) as f:
        queries = json.load(f)
    return docs, queries

def find_false_positives(rankings, relevant_set, rank_cutoff=FP_CUTOFF):
    """
    Retorna todos os pares (query_id, doc_id) em que um documento não
    relevante aparece até a posição `rank_cutoff` do ranking, ordenados
    pelo score.
    """
    rows = []
    for query_id, ranking in rankings.items():
        rel_set = relevant_set.get(query_id, set())
        for rank_pos, (doc_id, score) in enumerate(ranking[:rank_cutoff], start=1):
            if doc_id not in rel_set:
                rows.append({"query_id": query_id, "doc_id": doc_id,
                             "rank": rank_pos, "score": round(score, 4)})
    df = pd.DataFrame(rows)
    return df.sort_values("score", ascending=False) if not df.empty else df


def find_false_negatives(rankings, relevant_set, top_k=TOP_K):
    """
    Retorna todos os pares (query_id, doc_id) em que um documento relevante
    não aparece no Top-k, junto com sua posição real no ranking completo
    """
    rows = []
    for query_id, ranking in rankings.items():
        rel_set = relevant_set.get(query_id, set())
        if not rel_set:
            continue
        ranked_ids = [doc_id for doc_id, _ in ranking]
        top_k_ids = set(ranked_ids[:top_k])
        for doc_id in rel_set:
            if doc_id not in top_k_ids:
                real_rank = ranked_ids.index(doc_id) + 1 if doc_id in ranked_ids else None
                rows.append({"query_id": query_id, "doc_id": doc_id, "rank_real": real_rank})
    df = pd.DataFrame(rows)
    return df.sort_values("rank_real", na_position="last") if not df.empty else df


def term_diagnostics(query_tokens, doc_id, bm25_model):
    """TF/DF/IDF de cada termo da consulta para um documento específico."""
    freqs = bm25_model.doc_freqs.get(doc_id, {})
    rows = []
    for term in sorted(set(query_tokens)):
        rows.append({
            "term": term,
            "tf_in_doc": freqs.get(term, 0),
            "df_collection": bm25_model.df.get(term, 0),
            "idf_bm25": round(bm25_model._idf(term), 3),
        })
    return pd.DataFrame(rows)

def print_case(label, query_id, doc_id, bm25_model, queries,
                queries_raw_lookup, docs_raw_lookup, relevant_set):
    rel_set = relevant_set.get(query_id, set())
    print("\n" + "=" * 80)
    print(f"{label} | query_id={query_id} doc_id={doc_id}")
    print(f"consulta : {queries_raw_lookup.get(query_id, '')}")
    print(f"documento: {docs_raw_lookup.get(doc_id, '')}")
    print(f"relevante segundo qrels? {'SIM' if doc_id in rel_set else 'NÃO'}")

    doc_len = bm25_model.doc_tams.get(doc_id)
    avgdl = bm25_model.avgdl
    ratio = round(doc_len / avgdl, 2) if doc_len and avgdl else None
    print(f"tamanho do documento: {doc_len} tokens (avgdl={round(avgdl, 2)}, razão={ratio})")

    diag_df = term_diagnostics(queries[query_id], doc_id, bm25_model)
    print("diagnóstico por termo da consulta:")
    print(diag_df.to_string(index=False) if not diag_df.empty else "(sem termos)")


def main():
    docs, queries = load_processed(CONFIG_NAME)
    qrels_df = pd.read_csv(os.path.join(RAW_DIR, "qrels.csv"))
    relevant_set = relevant_sets(qrels_df)

    docs_raw_df = pd.read_csv(os.path.join(RAW_DIR, "docs.csv"))
    docs_raw_df["doc_id"] = docs_raw_df["doc_id"].astype(str)
    docs_raw_lookup = dict(zip(docs_raw_df["doc_id"], docs_raw_df["title"]))

    queries_raw_df = pd.read_csv(os.path.join(RAW_DIR, "queries.csv"))
    queries_raw_df["query_id"] = queries_raw_df["query_id"].astype(str)
    queries_raw_lookup = dict(zip(queries_raw_df["query_id"], queries_raw_df["text"]))

    # rank inteiro necessário para achar a posição real dos falsos negativos
    bm25_model = BM25Model(k1=1.2, b=0.75).fit(docs)
    rankings = bm25_model.rank_all_queries(queries, top_k=None)

    fp_df = find_false_positives(rankings, relevant_set)
    fn_df = find_false_negatives(rankings, relevant_set)

    os.makedirs(os.path.join(RESULTS_DIR, "agregados"), exist_ok=True)
    fp_df.to_csv(os.path.join(RESULTS_DIR, "agregados", f"false_positives_{CONFIG_NAME}.csv"), index=False)
    fn_df.to_csv(os.path.join(RESULTS_DIR, "agregados", f"false_negatives_{CONFIG_NAME}.csv"), index=False)

    print(f"Total de falsos positivos encontrados (rank <= {FP_CUTOFF}): {len(fp_df)}")
    print(f"Total de falsos negativos encontrados (relevante fora do Top-{TOP_K}): {len(fn_df)}")

    for _, row in fp_df.head(N_FP).iterrows():
        print_case("FALSO POSITIVO (não relevante no topo)", row["query_id"], row["doc_id"],
                    bm25_model, queries, queries_raw_lookup, docs_raw_lookup, relevant_set)

    for _, row in fn_df.head(N_FN).iterrows():
        print_case(f"FALSO NEGATIVO (relevante, rank real={row['rank_real']})",
                    row["query_id"], row["doc_id"], bm25_model, queries,
                    queries_raw_lookup, docs_raw_lookup, relevant_set)

if __name__ == "__main__":
    main()

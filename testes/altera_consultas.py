"""
Modificação manual de 5 consultas e comparação do Top-10 entre
a versão original e a modificada, para o Modelo Vetorial e o BM25.

Uso:
    python -m testes.altera_consultas
"""

import os
import json
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from src.modelo_vetorial import VectorModel
from src.modelo_probabilistico import BM25Model
from src.metrics import relevant_sets
from src.preprocessamento import nltk_resources, preprocess_text, config_params

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DIR = os.path.join(BASE_DIR, "dados", "processed")
RAW_DIR = os.path.join(BASE_DIR, "dados", "raw")
RESULTS_DIR = os.path.join(BASE_DIR, "resultados")

CONFIG_NAME = "ambas"
TOP_K = 10

MODIFIED_QUERIES = {
    "5": "what reaction rate model is applicable to hypersonic aerodynamic problems",       # sinônimo
    "9": "papers on heat transfer studies",                                                 # mais genérica
    "180": "how does scale height vary with altitude and temperature in an atmosphere?",    # acrescenta termos
    "30": "papers on flow visualization on slender conical wings at supersonic speeds",     # especifica
    "1": "what scaling laws must be satisfied when building aero-structural models of thermally loaded high-velocity airplanes", # muitos sinonimos
}

def load_processed(config_name):
    with open(os.path.join(PROCESSED_DIR, f"docs_{config_name}.json")) as f:
        docs = json.load(f)
    with open(os.path.join(PROCESSED_DIR, f"queries_{config_name}.json")) as f:
        queries = json.load(f)
    return docs, queries

def tokenize_modified_query(text, config_name, stop_words, stemmer):
    """Aplica o mesmo pipeline de pré-processamento usado em src/preprocessing.py,
    para que a consulta modificada seja comparável com as já processadas."""
    params = config_params(config_name)
    return preprocess_text(text, stop_words=stop_words, stemmer=stemmer, **params)


def compare_rankings(original_ranking, modified_ranking, relevant_set, k=TOP_K):
    """
    Compara o Top-k original vs. modificado: quais documentos entraram,
    saíram, e como cada um mudou de posição.
    """
    orig_ids = [doc_id for doc_id, _ in original_ranking[:k]]
    mod_ids = [doc_id for doc_id, _ in modified_ranking[:k]]

    orig_ranks = {doc_id: i + 1 for i, doc_id in enumerate(orig_ids)}
    mod_ranks = {doc_id: i + 1 for i, doc_id in enumerate(mod_ids)}

    entered = set(mod_ids) - set(orig_ids)
    left = set(orig_ids) - set(mod_ids)

    rows = []
    for doc_id in set(orig_ids) | set(mod_ids):
        status = "entrou" if doc_id in entered else "saiu" if doc_id in left else "manteve"
        rows.append({
            "doc_id": doc_id,
            "rank_original": orig_ranks.get(doc_id),
            "rank_modificado": mod_ranks.get(doc_id),
            "relevant": doc_id in relevant_set,
            "status": status,
        })
    df = pd.DataFrame(rows).sort_values(
        by=["rank_original", "rank_modificado"], na_position="last"
    )
    return df, len(entered), len(left)

def main():
    nltk_resources()
    stop_words = set(stopwords.words("english"))
    stemmer = SnowballStemmer("english")

    docs, queries = load_processed(CONFIG_NAME)
    qrels_df = pd.read_csv(os.path.join(RAW_DIR, "qrels.csv"))
    relevant_set = relevant_sets(qrels_df)

    queries_raw_df = pd.read_csv(os.path.join(RAW_DIR, "queries.csv"))
    queries_raw_df["query_id"] = queries_raw_df["query_id"].astype(str)
    queries_raw_lookup = dict(zip(queries_raw_df["query_id"], queries_raw_df["text"]))

    vector_model = VectorModel().fit(docs)
    bm25_model = BM25Model(k1=1.2, b=0.75).fit(docs)

    os.makedirs(os.path.join(RESULTS_DIR, "agregados"), exist_ok=True)
    all_rows = []

    for query_id, modified_text in MODIFIED_QUERIES.items():
        if query_id not in queries:
            print(f"query_id {query_id} não encontrado em data/processed; pulando.")
            continue

        original_tokens = queries[query_id]
        modified_tokens = tokenize_modified_query(modified_text, CONFIG_NAME, stop_words, stemmer)
        rel_set = relevant_set.get(query_id, set())

        print("\n" + "=" * 80)
        print(f"query_id={query_id}")
        print(f"original  : {queries_raw_lookup.get(query_id, '')}")
        print(f"  tokens  : {original_tokens}")
        print(f"modificada: {modified_text}")
        print(f"  tokens  : {modified_tokens}")

        for model_name, model in [("Vector", vector_model), ("BM25", bm25_model)]:
            orig_ranking = model.rank(original_tokens, top_k=TOP_K)
            mod_ranking = model.rank(modified_tokens, top_k=TOP_K)
            diff_df, n_entered, _ = compare_rankings(orig_ranking, mod_ranking, rel_set)
            diff_df["query_id"] = query_id
            diff_df["model"] = model_name
            all_rows.append(diff_df)

            print(f"\n  [{model_name}] documentos trocados no Top-{TOP_K}: {n_entered}")
            print(diff_df[["doc_id", "rank_original", "rank_modificado", "relevant", "status"]]
                  .to_string(index=False))

    if all_rows:
        full_df = pd.concat(all_rows, ignore_index=True)
        out_path = os.path.join(RESULTS_DIR, "agregados", f"consultas_alteradas_{CONFIG_NAME}.csv")
        full_df.to_csv(out_path, index=False)
        print(f"\nResultado completo salvo em {out_path}")


if __name__ == "__main__":
    main()
"""
Cálculo das métricas de avaliação exigidas: Precision@10, Recall@10, MAP.
Também inclui MRR como métrica adicional.

Convenção de relevância:
    - relevance >= 1    -> relevante
    - relevance == -1   -> não relevante
    - doc não julgado   -> não relevante
"""

from collections import defaultdict

def relevant_sets(qrels_df):
    """
    qrels_df: Dataframe com colunas query_id, doc_id, relevance
    Retorna dict {query_id: set(doc_ids relevantes)}
    """
    relevant = defaultdict(set)
    for _, row in qrels_df.iterrows():
        if row["relevance"] >= 1:
            relevant[str(row["query_id"])].add(str(row["doc_id"]))
    return dict(relevant)

def precision_at_k(ranked_doc_ids, relevant_set, k=10):
    top_k = ranked_doc_ids[:k]
    if len(top_k) == 0:
        return 0.0
    hits = sum(1 for d in top_k if d in relevant_set)
    return hits / len(top_k)

def recall_at_k(ranked_doc_ids, relevant_set, k=10):
    if len(relevant_set) == 0:
        return 0.0
    top_k = ranked_doc_ids[:k]
    hits = sum(1 for d in top_k if d in relevant_set)
    return hits / len(relevant_set)

def AP(ranked_doc_ids, relevant_set):
    """
    AP considerando todo o ranking, conforme definição padrão de MAP. 
    Para cada posição relevante encontrada, calcula a precisão até
    aquele ponto e tira a média.
    """
    if len(relevant_set) == 0:
        return 0.0

    hits = 0
    sum_precisions = 0.0
    for i, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant_set:
            hits += 1
            sum_precisions += hits / i

    if hits == 0:
        return 0.0
    return sum_precisions / len(relevant_set)

def RR(ranked_doc_ids, relevant_set):
    for i, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / i
    return 0.0

def avaliacao(ranking_dict, relevant_sets, k=10):
    """
    ranking_dict: {query_id: [(doc_id, score)]} ordenado
    relevant_sets: {query_id: set(doc_ids relevantes)}

    Retorna:
        rows: lista de dicts com métricas por consulta
        aggregated: dict com médias das métricas
    """
    rows = []
    for query_id, ranked in ranking_dict.items():
        ranked_doc_ids = [str(doc_id) for doc_id, _ in ranked]
        relevant_set = relevant_sets.get(str(query_id), set())

        p_at_k = precision_at_k(ranked_doc_ids, relevant_set, k)
        r_at_k = recall_at_k(ranked_doc_ids, relevant_set, k)
        ap = AP(ranked_doc_ids, relevant_set)
        rr = RR(ranked_doc_ids, relevant_set)

        rows.append({
            "query_id": query_id,
            f"precision@{k}": p_at_k,
            f"recall@{k}": r_at_k,
            "AP": ap,
            "RR": rr,
        })

    aggregated = {}
    if rows:
        for metric in [f"precision@{k}", f"recall@{k}", "AP", "RR"]:
            aggregated[metric] = sum(r[metric] for r in rows) / len(rows)
        aggregated["MAP"] = aggregated.pop("AP")
        aggregated["MRR"] = aggregated.pop("RR")

    return rows, aggregated
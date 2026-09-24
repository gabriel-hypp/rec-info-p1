"""
Baixa Cranfield via 'ir_datasets' e exporta docs, queries e qrels 
em arquivos dentro de dados/raw/, para que o resto não dependa 
diretamente do ir_datasets em toda execução.

Uso:
    python -m src.data_loader
"""

import os
import json
import pandas as pd
import ir_datasets

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "dados", "raw")

def load_dataset():
    """Carrega o dataset Cranfield via ir_datasets."""
    dataset = ir_datasets.load("cranfield")
    return dataset

def export_docs(dataset, out_path):
    """
    Exporta os documentos para um CSV com colunas: doc_id, title, text.
    Para o Cranfield, cada doc tem campos como title e text.
    """
    rows = []
    for doc in dataset.docs_iter():
        rows.append({
            "doc_id": doc.doc_id,
            "title": doc.title,
            "text": doc.text,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df

def export_queries(dataset, out_path):
    """Exporta as consultas para um CSV com colunas: query_id, text."""
    rows = []
    for query in dataset.queries_iter():
        rows.append({
            "query_id": query.query_id,
            "text": query.text,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df

def export_qrels(dataset, out_path):
    """
    Exporta os julgamentos de relevância para um CSV com colunas:
    query_id, doc_id, relevance.
    """
    rows = []
    for qrel in dataset.qrels_iter():
        rows.append({
            "query_id": qrel.query_id,
            "doc_id": qrel.doc_id,
            "relevance": qrel.relevance,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    dataset = load_dataset()

    export_docs(dataset, os.path.join(RAW_DIR, "docs.csv"))
    export_queries(dataset, os.path.join(RAW_DIR, "queries.csv"))
    export_qrels(dataset, os.path.join(RAW_DIR, "qrels.csv"))

    print("Dados brutos exportados com sucesso para dados/raw/.")

if __name__ == "__main__":
    main()
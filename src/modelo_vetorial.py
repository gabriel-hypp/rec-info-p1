"""
Implementa o Modelo Vetorial com TF-IDF e similaridade do cosseno para ranking.

Usa TfidfVectorizer do scikit-learn como apoio para a ponderação de termos.
"""

# Transforma coleção de tokens em matriz de recursos TF-IDF, abstraindo cálculos
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class VectorModel:
    def __init__(self):
        # DOcs já pré processados como listas de tokens
        # Usa "tokenizer" e "preprocessor" identidade no TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda tokens: tokens,
            preprocessor=lambda tokens: tokens,
            token_pattern=None,
            lowercase=False,
        )
        self.doc_ids = None
        self.doc_matrix = None # Matriz TF-IDF pós treino

    def fit(self, doc_tokens_dict):
        """
        doc_tokens_dict: dict {doc_id: [tokens]}
        Ajusta o vocabulário e o IDF a partir da coleção de documentos.
        """
        self.doc_ids = list(doc_tokens_dict.keys())
        docs_tokens = [doc_tokens_dict[d] for d in self.doc_ids]
        # Descobre palavras, calcula IDF de cada uma e converte documentos em matriz
        # Cada linha um documento, cada coluna uma palavra do vocabulário, valor = TF-IDF
        self.doc_matrix = self.vectorizer.fit_transform(docs_tokens)
        return self

    def rank(self, query_tokens, top_k=None):
        """
        Retorna uma lista ordenada [(doc_id, score)] para uma consulta
        já tokenizada, usando os mesmos IDFs ajustados em fit().
        """
        # Não usa fit para não recalcular IDF, apenas transforma consulta
        # Ignora palavras novas, para manter espaço vetorial igual
        query_vec = self.vectorizer.transform([query_tokens])
        # Calcula distância consulta e documentos, próx de 1 = mais termos
        # importantes em comum, proporcional ao tamanho do documento
        scores = cosine_similarity(query_vec, self.doc_matrix).flatten()

        # JUnta ids com scores e ordena decrescente pelo score, retorna top_k se solicitado
        ranked = sorted(zip(self.doc_ids, scores), key=lambda x: x[1], reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked

    def rank_all_queries(self, query_tokens_dict, top_k=None):
        """
        query_tokens_dict: dict {query_id: [tokens]}
        Retorna dict {query_id: [(doc_id, score), ...]}
        """
        results = {}
        for q_id, q_tokens in query_tokens_dict.items():
            results[q_id] = self.rank(q_tokens, top_k=top_k)
        return results

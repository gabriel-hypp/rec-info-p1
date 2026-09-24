"""
Implementação do modelo probabilístico BM25 seguindo os slides.
"""

import math
from collections import defaultdict, Counter

class BM25Model:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1  # Quanto menor, mais rápido score estagna com freq
        self.b = b    # 0 ignora tam documento, 1 penaliza mais

        self.doc_ids = None
        self.doc_freqs = None       # Counter(termo -> freq)
        self.doc_tams = None        # tamanho do documento
        self.avgdl = None
        self.df = None              # número de docs que contêm o termo
        self.N = None               # número total de docs
        self.idf_cache = {}

    def fit(self, doc_tokens_dict):
        """
        doc_tokens_dict: dict {doc_id: [tokens]}
        Constrói estruturas auxiliares (freqs, tamanhos, df)
        """
        self.doc_ids = list(doc_tokens_dict.keys())
        self.N = len(self.doc_ids)

        self.doc_freqs = {}
        self.doc_tams = {}
        df_counter = defaultdict(int)

        for doc_id in self.doc_ids:
            tokens = doc_tokens_dict[doc_id]
            freqs = Counter(tokens) # quantas vezes termo aparece no documento
            self.doc_freqs[doc_id] = freqs
            self.doc_tams[doc_id] = len(tokens)
            for term in freqs.keys():
                df_counter[term] += 1 # em quantos documentos o termo aparece

        self.df = dict(df_counter)
        self.avgdl = sum(self.doc_tams.values()) / self.N if self.N > 0 else 0
        self.idf_cache = {}
        return self

    def _idf(self, term):
        # Evita recalcular IDF
        if term in self.idf_cache:
            return self.idf_cache[term]
        n_qi = self.df.get(term, 0)
        idf = math.log((self.N - n_qi + 0.5) / (n_qi + 0.5) + 1)
        self.idf_cache[term] = idf
        return idf

    def _score(self, doc_id, query_tokens):
        score = 0.0
        doc_tam = self.doc_tams[doc_id]
        freqs = self.doc_freqs[doc_id]

        for term in query_tokens:
            f_qi_d = freqs.get(term, 0)
            # Se o termo n aparece no doc, não contribui score
            if f_qi_d == 0:
                continue
            idf = self._idf(term)
            num = f_qi_d * (self.k1 + 1)
            den = f_qi_d + self.k1 * (1 - self.b + self.b * doc_tam / self.avgdl)
            score += idf * (num / den)

        return score

    def rank(self, query_tokens, top_k=None):
        """Retorna [(doc_id, score)] ordenado por score decrescente."""
        scores = [(doc_id, self._score(doc_id, query_tokens)) for doc_id in self.doc_ids]
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked

    def rank_all_queries(self, query_tokens_dict, top_k=None):
        results = {}
        for q_id, q_tokens in query_tokens_dict.items():
            results[q_id] = self.rank(q_tokens, top_k=top_k)
        return results

    def set_params(self, k1, b):
        self.k1 = k1
        self.b = b
import math
import re
from collections import Counter

class BM25Retriever:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b

        self.postings = {}
        self.doc_lengths = []

        for index, doc in enumerate(documents):
            tokens = self.tokenize(doc["content"])
            self.doc_lengths.append(len(tokens))

            term_frequencies = Counter(tokens)

            for term, frequency in term_frequencies.items():
                if term not in self.postings:
                    self.postings[term] = []

                self.postings[term].append((index, frequency))

        if not self.doc_lengths:
            self.avg_length = 0
            self.length_norm = []
            return

        self.avg_length = sum(self.doc_lengths) / len(self.doc_lengths)
        self.length_norm = [self.k1 * (1 - self.b + self.b * length / self.avg_length) for length in self.doc_lengths]

    def tokenize(self, text):
        return re.findall(r"\w+", text.lower())

    def inverse_document_frequency(self, term):
        document_count = len(self.documents)

        if term not in self.postings:
            return 0.0

        term_document_count = len(self.postings[term])

        return math.log(1 + (document_count - term_document_count + 0.5) / (term_document_count + 0.5))

    def retrieve(self, query, top_k=3):
        query_terms = self.tokenize(query)
        scores = {}

        query_terms = set(query_terms)

        for term in query_terms:
            if term not in self.postings:
                continue

            idf = self.inverse_document_frequency(term)

            for index, doc_freq in self.postings[term]:
                tf_component = (doc_freq * (self.k1 + 1)) / (doc_freq + self.length_norm[index])

                score = idf * tf_component

                scores[index] = scores.get(index, 0.0) + score

        ranked_docs = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            self.documents[index]
            for index, _ in ranked_docs[:top_k]
        ]
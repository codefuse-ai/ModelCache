from sentence_transformers import SentenceTransformer

class MPNet_Base:
    def __init__(self):
        self.dimension = 768
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    def to_embeddings(self, *args, **kwargs):
        if not args:
            raise ValueError("No word provided for embedding.")
        embeddings = self.model.encode(args)
        return embeddings[0] if len(args) == 1 else embeddings

    def similarity(self, a, b):
        if not a or not b:
            raise ValueError("Both inputs must be non-empty for similarity calculation.")
        return self.model.similarity(a, b)

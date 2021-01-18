from sklearn.metrics.pairwise import cosine_similarity
from bert_serving.client import BertClient
from sentence_transformers import SentenceTransformer


MODELS = {
    "distil_roberta": SentenceTransformer("paraphrase-distilroberta-base-v1"),
    "bert": BertClient(),
}


class TransformerModel:
    def __init__(self, model="bert"):
        self.model = MODELS[model]

    def get_similarity(self, sentence1, sentence2):
        embedding1, embedding2 = (
            self.model.encode([s]) for s in [sentence1, sentence2]
        )
        # score is in this format: [[0.78]]
        score = cosine_similarity(embedding1, embedding2)
        # preserve only the float value
        return score[0][0]


import pandas as pd
import functools

from embedding_models import EmbeddingModel, MODELS
from utils import PROMPTS, get_valid_rows


def main():
    for model_name in MODELS:
        df = pd.read_csv("SENTANCES1/data1.csv")
        # filters out incomplete rows / rows with illegitimate responses
        df = df[df.apply(get_valid_rows, axis=1)]
        model = EmbeddingModel(model_name)

        for id, prompt in PROMPTS.items():
            # probably unnecessary (implicitly converted when using uncased BERT)
            df[id] = df[id].str.lower()
            # functools.partial returns a function with one of the argument specified
            df[id] = df[id].apply(functools.partial(model.get_similarity, prompt))
            print(id)

        df.to_csv("output/similarity_scores_{}.csv".format(model_name), index=False)


if __name__ == "__main__":
    main()

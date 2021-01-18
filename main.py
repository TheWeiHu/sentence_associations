import pandas as pd
import functools

from config import Config
from models import TransformerModel, MODELS


def get_valid_rows(row):
    # for simplicity, we only consider responses that:
    # 1) contain only legitimate answers (which would need at least 3 characters)
    # 2) are complete (answered all 5 prompts)
    for p in Config.prompts:
        entry = row[p]
        if not isinstance(entry, str) or len(entry) < 3:
            return False
    return True


def main():
    for model_name in MODELS:
        df = pd.read_csv("SENTANCES1/data1.csv")
        # filters out incomplete rows / rows with illegitimate responses
        df = df[df.apply(get_valid_rows, axis=1)]
        model = TransformerModel(model_name)
       
        for id, prompt in Config.prompts.items():
            # probably unnecessary (implicitly converted when using uncased BERT)
            df[id] = df[id].str.lower()
            # functools.partial returns a function with one of the argument specified
            df[id] = df[id].apply(functools.partial(model.get_similarity, prompt))
            print(id)

        df.to_csv("output/similarity_scores_{}.csv".format(model_name), index=False)


if __name__ == "__main__":
    main()

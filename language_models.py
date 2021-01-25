import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

"""
inspired by: https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-language-model-nlp-python-code/
"""


class LanguageModel:
    def __init__(self, name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.model.eval()

    def complete_sentence(self, text):
        n = 0
        while n < 100:
            text, finished = self.add_next_word(text)
            if finished:
                return text
            n += 1
        return text + "..."

    def add_next_word(self, text):
        indexed_tokens = self.tokenizer.encode(text)
        tokens_tensor = torch.tensor([indexed_tokens])

        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            predictions = outputs[0]

        predicted_index = torch.argmax(predictions[0, -1, :]).item()
        result = self.tokenizer.decode(indexed_tokens + [predicted_index])
        return result, result[-1] in ["?", ".", "!"]


if __name__ == "__main__":
    from utils import PROMPTS

    model = LanguageModel()
    for _, v in PROMPTS.items():
        print(model.complete_sentence(v))

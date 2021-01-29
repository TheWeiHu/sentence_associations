import string
from difflib import SequenceMatcher


def remove_punctuation(s):
    """
    most efficient according to:
        https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
    """
    table = str.maketrans(dict.fromkeys(string.punctuation))
    return s.translate(table)


def word_similarity(a, b):
    """
    a similarity metric between two given words
    e.g. "brother" would be very close to "brothers"
    """
    return SequenceMatcher(None, a, b).ratio()


def tokenize(s):
    """ 
    takes a sentence, converts it to lower case, removes punctuations and return it as a list of individual words.
    """
    return [x for x in remove_punctuation(s.lower()).split(" ") if x.strip()]


def calculate_overlap(s1, s2):
    """
    takes two sentences s1, s2, and indicates the percentage of words in s2 that are also in s1.
    """
    s1 = set(tokenize(s1))
    s2 = tokenize(s2)
    if not s2:
        return 1
    total = 0
    for s in s2:
        for t in s1:
            if word_similarity(t, s) >= 0.75:
                total += 1
                break
    return float(total / len(s2))


KAREN_PROMPTS = {
    "Love1": "When they talked about sex, I",
    "Love2": "After he made love to her, he",
    "Love3": "My sexual desires",
    "Love4": "Sexual intercourse",
    "Sex1": "A person who fall in love",
    "Sex2": "Love to me is",
    "Sex3": "When I think of marriage",
    "Sex4": "After a year of marriage, they",
}

TRAITS = ["EX", "NR", "OP", "AG", "CN"]
OPEN_PROMPTS = {
    "S1": "One should never trust",
    "S2": "The good life",
    "S3": "Brothers",
    "S4": "Sisters",
    "S5": "The beauty of",
    "S6": "Lacking everything but",
}


def get_valid_rows(row):
    # for simplicity, we only consider responses that:
    # 1) contain only legitimate answers (which would need at least 3 characters)
    # 2) are complete (answered all 5 OPEN_PROMPTS)
    for p in OPEN_PROMPTS:
        entry = row[p]
        if not isinstance(entry, str) or len(entry) < 3:
            return False
    return True


if __name__ == "__main__":
    print(tokenize("The good life is..."))
    print(word_similarity("brothers", "brothersdd"))
    print(calculate_overlap("Brothers...", ", oh brothers."))

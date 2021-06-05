import fasttext.util
from gensim.utils import simple_preprocess


def tokenizer(string):
    processed = simple_preprocess(string)

    print(processed)

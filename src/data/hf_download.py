from datasets import load_dataset

data_dir = '../../../data'
names = ['amazon_reviews_multi']
langs = ['en']

for name, lang in zip(names, langs):
    d = load_dataset(name, lang, cache_dir=data_dir)

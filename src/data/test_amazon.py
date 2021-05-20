import os

from datasets import load_dataset
from transformers import AutoTokenizer
import fasttext.util
from gensim.utils import simple_preprocess
import torch
from torch.utils.data import DataLoader

data_dir = '../../../data'
embedding_file = '../../../data/fasttext/cc.en.300.bin'

# check fasttext embedding_file and download
if not os.path.exists(embedding_file):
    print('downloading')
    cwd = os.getcwd()
    os.chdir(data_dir+'/fasttext')
    fasttext.util.download_model('en', if_exists='ignore')
    os.chdir(cwd)
else:
    word_vecs = fasttext.load_model(embedding_file)
    print('testing fasttext:')
    print('hello:', word_vecs['hello'].shape)

def tokenizer(text, max_len=20):
    embed_list = []
    for i, word in enumerate(simple_preprocess(text)):
        if i >= max_len:
            break
        embed_list.append(word_vecs[word])
    while i < max_len-1:
        embed_list.append(word_vecs['nan'])
        i += 1

    return embed_list
    
    
# load amazon_dataset
d = load_dataset('amazon_reviews_multi', 'en', cache_dir=data_dir, split='train')
d = d.remove_columns(['language', 'product_category', 'product_id', 'review_body', 'review_id', 'reviewer_id'])
d = d.select(range(32))
print('testing dataset...')
print(d[0]['review_title'], d[0]['stars'])
print('\n\n')

dataset = d.map(lambda x: {
    'data': torch.Tensor(tokenizer(x['review_title'])),
    'label': x['stars']
}, batched=False)
dataset = dataset.remove_columns(['review_title', 'stars'])
#dataset.set_format(type='torch', columns=['data', 'label'])
print(dataset[0].keys())
print(type(dataset[0]['data']), len(dataset[0]['data']))
# print(type(dataset[0]['data'][0]), len(dataset[0]['data'][0]))
#exit()
dataloader = DataLoader(dataset, batch_size=32)

for data in dataloader:
    print('label_shape: ', data['label'].shape)
    print(type(data['data']), len(data['data']))

    break

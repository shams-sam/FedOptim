import fasttext.util
from torch.utils.data import Dataset

from common.text_processing import text_processing

class AmazonDataset(Dataset):
    def __init__(self, data_dir, embedding_file, split):
        word_vecs = fasttext.load_model(embedding_file)
        dataset = load_dataset('amazon_reviews_multi', 'en',
                               cache_dir=data_dir, split=split)

    def tokenize(text):
        

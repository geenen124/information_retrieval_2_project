import h5py
import torch
from torch.utils.data import Dataset

from data_util.bin_files_to_h5 import convert_bin_files_to_h5, h5_db_filepath
from data_util import data
from data_util.batcher import Example


class DailyMailDataset(Dataset):
    """Loads the CNN-DailyMail Dataset"""
    def __init__(self, file_type, vocab):
        # First generate h5py files if they don't exist
        convert_bin_files_to_h5()

        self.db = h5py.File(h5_db_filepath(file_type))
        self.vocabulary = vocab

    def __del__(self):
        self.db.close()

    def __len__(self):
        return self.db["articles"].shape[0]

    def __getitem__(self, index):
        article = self.db["articles"][index]
        abstract = self.db["abstracts"][index]

        # use existing Example from threaded Batcher code
        abstract_sentences = [sent.strip() for sent in data.abstract2sents(abstract)] # Use the <s> and </s> tags in abstract to get a list of sentences.
        example = Example(article, abstract_sentences, self.vocabulary) # Process into an Example.

        return example

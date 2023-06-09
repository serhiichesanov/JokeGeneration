from torch.utils.data import Dataset
import torch
from transformers import GPT2Tokenizer
import pandas as pd


class JokeDataset(Dataset):
    def __init__(self, jokes):
        self.jokes = jokes

    def __len__(self):
        return len(self.jokes)

    def __getitem__(self, idx):
        return self.jokes[idx]


def load_and_process_data():
    df = pd.read_csv('./data/shortjokes.csv')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    jokes = df['Joke'].tolist()
    jokes_tokenized = [torch.tensor(tokenizer.encode(joke)) for joke in jokes]
    return jokes_tokenized, tokenizer

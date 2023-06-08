import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import pandas as pd

class JokeGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers):
        super(JokeGenerator, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        x = self.embed(x)
        output, _ = self.rnn(x)
        output = self.fc(output)
        return output
    
class JokeDataset(Dataset):
    def __init__(self, jokes):
        self.jokes = jokes

    def __len__(self):
        return len(self.jokes)

    def __getitem__(self, idx):
        return self.jokes[idx]
    
def load_and_process_data():
    df = pd.read_csv('data/shortjokes.csv')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    jokes = df['Joke'].tolist()
    jokes_tokenized = [torch.tensor(tokenizer.encode(joke)) for joke in jokes]
    return jokes_tokenized, tokenizer

jokes_tokenized, tokenizer = load_and_process_data()
tokenizer.pad_token = tokenizer.eos_token

dataset = JokeDataset(jokes_tokenized)
loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda batch: pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id))

vocab_size = tokenizer.vocab_size
embedding_dim = 256
hidden_dim = 512
n_layers = 2

model = JokeGenerator(vocab_size, embedding_dim, hidden_dim, n_layers)
model.load_state_dict(torch.load('trained_model/joke_generator_model_epoch_2.pt'))

model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Генерація жарту
def generate_joke(start_string, length=50):
    model.eval()

    start_tokens = tokenizer.encode(start_string)
    num_generated = 0
    generated_tokens = start_tokens

    with torch.no_grad():
        for _ in range(length):
            inputs = torch.tensor([generated_tokens]).to(device)
            outputs = model(inputs)
            predictions = outputs[0, -1, :]
            predicted_token = torch.argmax(predictions)
            generated_tokens.append(predicted_token.item())
            num_generated += 1
            if num_generated >= length or predicted_token.item() == tokenizer.eos_token_id:
                break

    generated_joke = tokenizer.decode(generated_tokens).replace("<|endoftext|>", "")
    return generated_joke

start_string = "Chicken"
print(generate_joke(start_string))

start_string = "Doctor"
print(generate_joke(start_string))

start_string = "Animal"
print(generate_joke(start_string))

start_string = "Cowboy"
print(generate_joke(start_string))

start_string = "Software engineer"
print(generate_joke(start_string))

start_string = "Headphones"
print(generate_joke(start_string))

start_string = "Why did the chicken cross the road?"
print(generate_joke(start_string))
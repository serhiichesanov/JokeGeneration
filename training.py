import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
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
epochs = 5

model = JokeGenerator(vocab_size, embedding_dim, hidden_dim, n_layers)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()


epochs = 5
for epoch in range(epochs):
    for batch in loader:
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        outputs = model(inputs)

        loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{epochs} Loss: {loss.item()}')

    torch.save(model.state_dict(), f'trained_model/joke_generator_model_epoch_{epoch+1}.pt')
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from classes.joke_generator import JokeGenerator
from classes.dataset import load_and_process_data, JokeDataset


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
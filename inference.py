import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from classes.joke_generator import JokeGenerator
from classes.dataset import load_and_process_data, JokeDataset

jokes_tokenized, tokenizer = load_and_process_data()
tokenizer.pad_token = tokenizer.eos_token

dataset = JokeDataset(jokes_tokenized)
loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda batch: pad_sequence(batch, batch_first=True,
                                                                                                padding_value=tokenizer.pad_token_id))

vocab_size = tokenizer.vocab_size
embedding_dim = 256
hidden_dim = 512
n_layers = 2

model = JokeGenerator(vocab_size, embedding_dim, hidden_dim, n_layers)
model.load_state_dict(torch.load('trained_model/joke_generator_model_epoch_2.pt'))

model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

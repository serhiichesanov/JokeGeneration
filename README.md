# JokeGeneration
This project focuses on training a joke generation model using GPT-2 tokenizer and GRU-based model. The code provided consists of three parts: training the joke generation model, using the trained model to generate jokes and running the telegram bot
## Project Description
The project aims to develop a joke generation model using an RNN. The model is based on the JokeGenerator class, which consists of an embedding layer, a GRU (Gated Recurrent Unit) layer, and a linear layer. The model is trained on a dataset of short jokes and can generate jokes based on user prompts.
## Data Preprocessing
The data preprocessing steps are handled in the load_and_process_data function in the training code. The function reads the "shortjokes.csv" dataset using pandas and tokenizes each joke using the GPT2Tokenizer from the Transformers library. The tokenized jokes are then returned as a list.
## Model Architecture
The joke generation model is defined in the JokeGenerator class. It consists of an embedding layer, a GRU layer, and a linear layer. The model takes tokenized input sequences and predicts the next token in the sequence
## Training the Model
The training code uses the JokeDataset class and the DataLoader to load the tokenized jokes as a PyTorch dataset and create data batches. It then iterates over the data batches and performs forward and backward passes to train the model using the Adam optimizer and the CrossEntropyLoss criterion. After each epoch, the model is saved to a file.
## Generating Jokes
The joke generation code uses the trained model to generate jokes based on user prompts. The generate_joke function takes a starting string and a desired length as input. It generates jokes by repeatedly predicting the next token in the sequence and stopping either when the desired length is reached or when the end-of-sequence token is generated.
## To run this program:
1. Clone the github repo onto your local machine: -git clone https://github.com/serhiichesanov/JokeGeneration.git
2. Install the necessary requirements into a virtual environment by running the command: 'pip install -r Requirements.txt'
3. Run 'training.py' to train your neural network (Due to the size limit, we are unable to publish already trained nn on github)
4. Create your own Teleram bot, get its token and replace "-" with it in 'app.py'
5. Run bot: 'python app.py'
## Example of generating jokes
Exapmle 1:

![image](https://github.com/serhiichesanov/JokeGeneration/assets/91079312/4dfc6fb0-fe5b-4789-9ca9-9e9b42a1ed74)
Example 2:

![image](https://github.com/serhiichesanov/JokeGeneration/assets/91079312/2dfe01f1-3339-4c37-acb9-9eecfc7ed9a4)

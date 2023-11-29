from transformers import GPT2Tokenizer, GPT2Model
import torch

class GPT2Encoder:
    def __init__(self):
        # Load pre-trained model tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # Load pre-trained model
        self.model = GPT2Model.from_pretrained('gpt2')
        # Put the model in "evaluation" mode
        self.model.eval()

    def encode(self, text):
        # Encode text
        encoded_input = self.tokenizer(text, return_tensors='pt')
        # Forward pass, calculate embeddings
        with torch.no_grad():
            output = self.model(**encoded_input)
        return output.last_hidden_state

def index(text):
    encoder = GPT2Encoder()
    return encoder.encode(text)

if __name__ == '__main__':
    sample_text = "Your sample text goes here"
    embeddings = index(sample_text)
    print(embeddings)

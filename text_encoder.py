from transformers import AutoTokenizer, AutoModel
import configparser
import torch

# Load model from a configuration file
config = configparser.ConfigParser()
config.read("config.ini")

# Initialize the tokenizer and model
MODEL = config.get("EncodingModel", "MODEL")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL)


def encode_texts(texts: list[str]):
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Get sentence embeddings using mean pooling of last hidden state
    with torch.no_grad():
        outputs = model(**encoded)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings


def encode_with_sliding_context(sentence: str, context_length: int):
    words = sentence.split()
    embeddings = []

    # Slide the context window over the sentence
    for i in range(len(words) - context_length + 1):
        # Take 'context_length' words starting from index i
        context = " ".join(words[i : i + context_length])
        embedding = encode_texts([context])
        embeddings.append(embedding.squeeze().tolist())

    return embeddings


def encode_using_multiple_context_sizes(sentence: str, context_lengths: list[str]):
    result = []
    for context_length in context_lengths:
        result.append(encode_with_sliding_context(sentence, context_length))
    return result

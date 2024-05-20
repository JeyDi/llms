from typing import List

import openai
import torch
from fastembed import TextEmbedding
from transformers import AutoModel, AutoTokenizer

from experiments.config import settings

DEVICE = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")


def generation_fastembed(input_data: List[str]):
    """Generate fast embeddings for the input data using the specified model.

    Parameters:
    - input_data: List of strings to generate embeddings for.

    Returns:
    - List of embeddings generated from the input data.
    """
    # This will trigger the model download and initialization
    embedding_model = TextEmbedding(model_name="BAAI/bge-large-en-v1.5")

    # embeddings_generator = embedding_model.embed(input_data)  # reminder this is a generator
    embeddings_list = list(embedding_model.embed(input_data))
    # you can also convert the generator to a list, and that to a numpy array
    len(embeddings_list[0])  # Vector of 384 dimensions

    return embeddings_list


def generation_hugging_face(input_data: List[str]):
    model_name = "allenai/specter2_base"
    model = AutoModel.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    # load model and tokenizer
    # Load model and tokenizer
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize the input data
    encoded_input = tokenizer(input_data, return_tensors="pt", padding=True, truncation=True)

    # Move the encoded input to the device (GPU if available)
    encoded_input = encoded_input.to(DEVICE)

    # Generate the embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Extract the embeddings from the [CLS] tokens
    embeddings = model_output.last_hidden_state[:, 0, :]

    # Convert the tensor to a list of lists of floats
    return embeddings.cpu().numpy().tolist()


def generate_openai(input_data: List[str], model: str = "text-embedding-ada-002"):
    embeddings = []
    for text in input_data:
        response = openai.Embedding.create(model=model, input=text)
        embedding_vector = response["data"][0]["embedding"]
        embeddings.append(embedding_vector)

    return embeddings


if __name__ == "__main__":
    result = generation_hugging_face(["This is a test sentence", "This is another test sentence"])
    # print(TextEmbedding.list_supported_models())
    # result = generation_fastembed(["This is a test sentence", "This is another test sentence"])
    # result = generate_openai(["This is a test sentence", "This is another test sentence"])
    print("\nGeneration complete.")
    print(f"Embedding dimension: {len(result[0])}")
    print(f"\nExample:{result[0]}")

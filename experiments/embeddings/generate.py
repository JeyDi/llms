from typing import List

import torch
from fastembed import TextEmbedding
from transformers import AutoModel

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

    embeddings_generator = embedding_model.embed(input_data)  # reminder this is a generator
    embeddings_list = list(embedding_model.embed(input_data))
    # you can also convert the generator to a list, and that to a numpy array
    len(embeddings_list[0])  # Vector of 384 dimensions

    return embeddings_list


def generation_hugging_face(input_data: str):
    model_name = "allenai/specter2_base"
    model = AutoModel.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    # load model and tokenizer
    # tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')

    # load base model
    # model = AutoAdapterModel.from_pretrained('allenai/specter2_base')

    # load the adapter(s) as per the required task, provide an identifier for the adapter in load_as argument and activate it
    # model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)

    # Tokenizza i dati di input
    encoded_input = model.tokenizer(input_data, return_tensors="pt", padding=True, truncation=True)

    # Sposta gli input su GPU (se disponibile)
    encoded_input = encoded_input.to("cuda")

    # Genera gli embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Estrai gli embeddings dai token [CLS]
    sentence_embeddings = model_output.last_hidden_state[:, 0, :]

    # Normalizza gli embeddings utilizzando la norma L2
    normalized_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

    return normalized_embeddings


if __name__ == "__main__":
    # generation_hugging_face(["This is a test sentence", "This is another test sentence"])
    # print(TextEmbedding.list_supported_models())
    result = generation_fastembed(["This is a test sentence", "This is another test sentence"])
    print(result)

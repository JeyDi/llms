from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct

from experiments.config import settings
from experiments.datasets.arxiv import load_parquet_dataset
from experiments.embeddings.generate import generate_openai
from experiments.text.utils import clean_text

CLIENT = QdrantClient(
    url=settings.QDRANT_HOST,
    port=settings.QDRANT_PORT,
    api_key=settings.QDRANT_API_KEY,
)  # Adjust host and port as needed


def create_collection(collection_name: str = settings.QDRANT_COLLECTION_NAME, vector_size: int = 1536):
    CLIENT.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )
    print(f"Collection {collection_name} created")


def insert_embeddings_to_qdrant(
    embeddings: List[str], payload: dict = None, collection_name: str = settings.QDRANT_COLLECTION_NAME
):

    for index, embedding in enumerate(embeddings):
        attributes = payload[index]
        point = PointStruct(
            id=index,  # Unique identifier for the point
            vector=embedding,
            payload=attributes if attributes else {"example_field": "example_value"},  # Optional payload
        )
        CLIENT.upsert(collection_name=collection_name, points=[point])
        print(f"Point {index} inserted")


def create_index(collection_name: str = settings.QDRANT_COLLECTION_NAME, index: dict = None):

    if not index:
        index = {"id": "integer", "doi": "text"}

    for index_name, schema in index.items():
        CLIENT.create_payload_index(
            collection_name=collection_name,
            field_name=index_name,
            field_schema=schema,
        )
        print(f"Index {index_name} created")


if __name__ == "__main__":
    # Load the dataframe
    dataset = load_parquet_dataset("./data/arxiv/dataset/dataset.parquet")
    print(dataset.shape)
    print(dataset.head())
    # Subset the DataFrame
    subset_df = dataset.head(settings.MAX_ARTICLES)
    # Combine title and abstract
    combined_cleaned = [clean_text(f"{row['title']} {row['abstract']}") for index, row in subset_df.iterrows()]
    subset_df = subset_df.drop(columns=["title", "abstract"])
    # Generate the embedding
    embeddings = generate_openai(combined_cleaned)
    # Create a collection
    create_collection(vector_size=len(embeddings[0]))
    # Insert a new point
    insert_embeddings_to_qdrant(
        collection_name=settings.COLLECTION_NAME, embeddings=embeddings, payload=subset_df.to_dict(orient="records")
    )
    # Generate the index over the collection for the payload
    create_index()

    print("Done!")

import os
import json
from datasets import load_dataset
import pandas as pd


def manual_import(dataset_raw: str = "./data/arxiv/raw"):
    """Manual import the arxiv dataset from the json file.

    Args:
        dataset_raw (str, optional): the folder there the json is written and located. Defaults to "./data/arxiv/raw".

    Returns:
        pd.Dataframe: pandas dataframe
    """
    cols = ["id", "title", "abstract", "categories"]
    data = []
    file_name = os.path.join(dataset_raw, "arxiv-metadata-oai-snapshot.json")

    with open(file_name, encoding="latin-1") as f:
        for line in f:
            doc = json.loads(line)
            lst = [doc["id"], doc["title"], doc["abstract"], doc["categories"]]
            data.append(lst)

    df_data = pd.DataFrame(data=data, columns=cols)

    print(df_data.shape)

    df_data.head()
    return df_data


def load_arxiv(output_folder: str = "./data/arxiv/dataset", dataset_raw: str = "./data/arxiv/raw"):
    """Load arxiv dataset from hugging face and save it as parquet.

    Warning: to download the dataset you need to go to Kaggle: https://www.kaggle.com/datasets/Cornell-University/arxiv?resource=download
    and download manually the dataset.
    Then unzip the dataset into the dataset_raw folder.
    In this function we will load the dataset from the dataset_raw folder in hugging face datasets format.

    Args:
        output_folder (str, optional): _description_. Defaults to "./data/arxiv/dataset".
        dataset_raw (str, optional): _description_. Defaults to "./data/arxiv/raw".

    Returns:
        datasets: arxiv dataset un hugging face format
    """
    # load the arXiv dataset from hugging face
    dataset = load_dataset("arxiv_dataset", data_dir=dataset_raw, trust_remote_code=True)

    # explore the dataset
    print(dataset)

    # access the dataset data
    papers = dataset["train"]
    for paper in papers:
        print(paper["title"], paper["abstract"])

    # transform to parquet
    papers_df = pd.DataFrame(papers)

    output_file = os.path.join(output_folder, dataset.parquet)
    papers_df.to_parquet(output_file)

    return dataset


if __name__ == "__main__":
    print("Hello from arxiv.py!")
    dataset = manual_import()
    # dataset = load_arxiv() #using the hugging face dataset is not working

import json
import os

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def manual_import(dataset_raw: str = "./data/arxiv/raw"):
    """Manual import the arxiv dataset from the json file.

    Columns:
        - id: ArXiv ID (can be used to access the paper, see below)
        - submitter: Who submitted the paper
        - authors: Authors of the paper
        - title: Title of the paper
        - comments: Additional info, such as number of pages and figures
        - journal-ref: Information about the journal the paper was published in
        - doi: [https://www.doi.org](Digital Object Identifier)
        - abstract: The abstract of the paper
        - categories: Categories / tags in the ArXiv system
        - versions: A version history

        You can access each paper directly on ArXiv using these links:

            https://arxiv.org/abs/{id}: Page for this paper including its abstract and further links
            https://arxiv.org/pdf/{id}: Direct link to download the PDF

    Args:
        dataset_raw (str, optional): the folder there the json is written and located. Defaults to "./data/arxiv/raw".

    Returns:
        pd.Dataframe: pandas dataframe
    """
    cols = ["id", "title", "abstract", "categories"]
    data = []
    file_name = os.path.join(dataset_raw, "arxiv-metadata-oai-snapshot.json")

    with open(file_name, encoding="latin-1") as f:
        total_lines = sum(1 for _ in f)
        f.seek(0)  # Reset the file pointer to the beginning
        for line in tqdm(f, total=total_lines):
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

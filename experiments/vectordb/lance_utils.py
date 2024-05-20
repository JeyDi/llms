import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

from experiments.config import settings
from experiments.embeddings.generate import generate_openai

db = lancedb.connect("/tmp/db")
func = get_registry().get("openai").create(name="text-embedding-ada-002")


class Words(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()


table = db.create_table("words", schema=Words, mode="overwrite")
table.add([{"text": "hello world"}, {"text": "goodbye world"}])

query = "greetings"
actual = table.search(query).limit(1).to_pydantic(Words)[0]
print(actual.text)


if __name__ == "__main__":

    embedding = generate_openai(["This is a test"])

from experiments.config import settings
from huggingface_hub import snapshot_download
import os

mistral_models_path = settings.MISTRAL_MODEL_PATH

if not os.path.exists(mistral_models_path):
    os.makedirs(mistral_models_path)

settings._authenticate_hugging_face()

if __name__ == "__main__":
    snapshot_download(
        repo_id="mistralai/Mistral-7B-v0.3",
        local_dir=mistral_models_path,
        force_download=False,
    )

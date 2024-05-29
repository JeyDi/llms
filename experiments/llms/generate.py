import torch
import typer
from experiments.config import settings
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

app = typer.Typer()


@app.command()
def chat():
    # Load the model and tokenizer
    model_name = "mistralai/Mistral-7B-v0.3"  # Replace with the desired LLaMA model
    # model_name = settings.MISTRAL_MODEL_PATH
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Move the model to the GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    while True:
        # Ask for user input
        input_text = typer.prompt("Enter your text (or 'exit' to stop):")
        if input_text.lower() == "exit":
            break

        # Encode input text
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        # Generate text
        with torch.no_grad():
            output = model.generate(input_ids, max_length=50, num_return_sequences=1)

        # Decode and print the output text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(generated_text)


if __name__ == "__main__":
    app()

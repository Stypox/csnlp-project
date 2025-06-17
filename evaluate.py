
import argparse
import math

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_dtype(torch.bfloat16)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")
torch.set_default_device(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="The repo_id or the path of the model")
    parser.add_argument("--dataset", type=str, default="allenai/c4", help="The repo_id of the dataset")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map=device,
        trust_remote_code=True,
    )

    def tokenize(example):
        return tokenizer(
            example["text"],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=args.max_length+1
        )

    ds = load_dataset(args.dataset, "en", split="validation", streaming=True)
    ds = ds.map(tokenize).with_format("torch")
    dataloader = DataLoader(ds, batch_size=args.batch_size)
    iter_dataloader = iter(dataloader)

    total_perplexity = 0
    count = 0
    for epoch in range(args.epochs):
        input_ids = next(iter_dataloader)["input_ids"][0]
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
        total_perplexity += math.exp(loss.item())
        count += 1

        print(f"Perplexity: {total_perplexity / count:e} (step {epoch+1: 4}/{args.epochs})", end="\r")
    print()
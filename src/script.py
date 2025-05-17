# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM
from datasets import load_dataset
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

torch.set_default_dtype(torch.bfloat16)
device = torch.device('cuda:0')
torch.set_default_device(device)

# this is just a debugging function to get the first value from a Tensor
def topleft(a):
    if a is None:
        return None
    elif len(a.shape) == 0:
        return a.item()
    elif len(a.shape) == 1:
        return a.to(torch.float32).cpu().clone().detach().numpy()[0]
    else:
        return topleft(a[0])

class Trainer:
    def __init__(
        self,
        model: str,
        layers: list[int],
        regularization_lambda: float,
        learning_rate: float,
    ):
        self.layers: list[int] = layers
        self.regularization_lambda: float = regularization_lambda
        self.learning_rate: float = learning_rate
        self.loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss()

        self.model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(model).to(device)

        self.enable_only_layer_gradients()

        self.optimizer: torch.optim.SGD = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

    def is_layer_enabled(self, layer_name: str):
        return layer_name.startswith(f"model.layers.") and int(layer_name.split(".")[2]) in self.layers

    def debug_layer(self):
        for name, param in self.model.named_parameters():
            if self.is_layer_enabled(name):
                print(name, topleft(param.grad))
            else:
                assert param.grad is None

    def enable_only_layer_gradients(self):
        for name, param in self.model.named_parameters():
            if self.is_layer_enabled(name):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def get_layer_grad_l2(self):
        res = 0
        for name, param in self.model.named_parameters():
            if self.is_layer_enabled(name):
                res += self.regularization_lambda * torch.norm(param.grad, p=2)
        return res

    def train_on_batch(self, input: torch.LongTensor, target: torch.LongTensor):
        # TODO what is res[1]??
        res = self.model(input)
        logits = res[0].reshape(-1, res[0].size(-1))

        self.optimizer.zero_grad()
        loss = self.loss_fn(logits, target)
        loss.backward(retain_graph=True)
        loss2 = loss.clone() + self.get_layer_grad_l2()
        self.optimizer.zero_grad()
        loss2.backward()
        self.optimizer.step()

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a model on C4 dataset")
    parser.add_argument("--layers", type=int, nargs='+', default=[19], help="Layers to train")
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--reg_lambda", type=float, default=1e-2, help="Regularization lambda")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="The repo_id of the model")
    args = parser.parse_args()

    # Prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    def tokenize(example):
        return tokenizer(example["text"], return_tensors="pt", truncation=True, padding="max_length", max_length=args.max_length)

    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    ds = ds.shuffle().map(tokenize).with_format("torch")

    dataloader = DataLoader(ds, batch_size=args.batch_size)

    # Prepare model
    trainer = Trainer(
        model=args.model,
        layers=args.layers,
        regularization_lambda=args.reg_lambda,
        learning_rate=args.learning_rate,
    )

    # Do training
    print(f"Training on C4 dataset for {args.steps} steps...")
    for step, batch in enumerate(dataloader):
        inputs = batch['input_ids'].to(device).squeeze(1)
        labels = inputs.clone()
        labels = labels.reshape(-1)

        trainer.train_on_batch(inputs, labels)

        print(f"Step: {step}/{args.steps}", end="\r")
        if step % 10 == 0:
            print()
            trainer.debug_layer()

    print("Training complete!")

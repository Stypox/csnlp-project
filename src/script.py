# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

import torch
from torch import nn
from transformers import LlamaConfig, AutoTokenizer
from model import LlamaForCausalLM
from datasets import load_dataset
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
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

class Traning:
    def __init__(
        self,
        layers: list[int],
        regularization_lambda: float,
        learning_rate: float,
    ):

        self.layers: list[int] = layers
        self.regularization_lambda: float = regularization_lambda
        self.learning_rate: float = learning_rate
        self.loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss()

        path = hf_hub_download(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", filename="model.safetensors")
        tensors = load_file(path, device="cuda:0")

        self.model: LlamaForCausalLM = LlamaForCausalLM(
            LlamaConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        ).to(device) # pyright: ignore[reportArgumentType]
        self.model.load_state_dict(tensors)

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

def load_c4_batch(tokenizer, batch_size=8, max_length=2048, split="train"):
    """
    Load a batch of data from the C4 dataset.
    
    Args:
        tokenizer: Tokenizer to use
        batch_size: Batch size
        max_length: Maximum sequence length
        split: Dataset split to use
        
    Returns:
        Tuple of (input_ids, labels) tensors
    """
    # Load C4 dataset
    dataset = load_dataset("c4", "en", split=split, streaming=True)
    batch_texts = []
    
    # Collect batch_size examples
    for item in dataset:
        batch_texts.append(item['text'])
        if len(batch_texts) == batch_size:
            break
    
    # Tokenize
    encodings = tokenizer(
        batch_texts,
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors='pt'
    )
    
    # Get input_ids and labels
    input_ids = encodings['input_ids'].to(device)
    labels = input_ids.clone()  # For causal language modeling
    
    return input_ids, labels

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a model on C4 dataset")
    parser.add_argument("--layers", type=int, nargs='+', default=[19], help="Layers to train")
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--reg_lambda", type=float, default=1e-2, help="Regularization lambda")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
    args = parser.parse_args()
    
    # Initialize the model trainer
    t = Traning(
        layers=args.layers,
        regularization_lambda=args.reg_lambda,
        learning_rate=args.learning_rate,
    )
    
    # Initialize tokenizer for C4 dataset
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    print(f"Training on C4 dataset for {args.steps} steps...")
    
    # Training loop with C4 dataset
    for step in range(args.steps):
        # Get real data from C4 instead of dummy tensors
        input_ids, labels = load_c4_batch(
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length
        )
        
        # Train using the same method as before
        t.train_on_batch(input_ids, labels)
        
        # Debug every 100 steps
        if step % 100 == 0:
            print(f"Step: {step}/{args.steps}")
            t.debug_layer()
    
    print("Training complete!")

# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from datasets import load_dataset
import pandas as pd
import argparse
import signal
import os
import re
import time
import random

torch.set_default_dtype(torch.bfloat16)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")
torch.set_default_device(device)
torch.backends.cuda.enable_flash_sdp(False)

# to capture ctrl+c during training
ctrl_c = False
def handler(_signum, _frame):
    global ctrl_c
    if ctrl_c:
        exit(1)
    ctrl_c = True

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

# just for formatting floats
def format_exp(val):
    s = "{:e}".format(val)
    s = re.sub(r"e([+-])0+([0-9])", r"e\1\2", s)
    s = re.sub(r"\.?0+e", "e", s)
    s = re.sub(r"e([+-])0", "", s)
    return s

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
        self.model_name: str = model.split("/")[1]

        self.model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype="auto",
            device_map=device,
            trust_remote_code=True,
        )

        self.enable_only_layer_gradients()

        self.optimizer: torch.optim.SGD = torch.optim.SGD([p for p in self.model.parameters() if p.requires_grad], lr=learning_rate)

    def is_layer_enabled(self, layer_name: str):
        return layer_name.startswith("model.layers.") and int(layer_name.split(".")[2]) in self.layers

    def debug_layer(self):
        for name, param in self.model.named_parameters():
            if self.is_layer_enabled(name):
                print(name, f"{topleft(param.grad):e}")
            else:
                assert param.grad is None

    def enable_only_layer_gradients(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = self.is_layer_enabled(name)

    def train_on_batch(
        self,
        input: torch.LongTensor,
        target: torch.LongTensor,
        attention_mask: torch.LongTensor,
        is_harmful: bool,
    ):
        # TODO what is res[1]??
        res = self.model(input_ids=input, attention_mask=attention_mask)
        logits = res.logits.view(-1, res.logits.size(-1))

        loss = self.loss_fn(logits, target)
        orig_crossentropyloss = loss.item()
        grads = torch.autograd.grad(loss, [p for n,p in self.model.named_parameters() if self.is_layer_enabled(n)], create_graph=True)

        # add L2 gradient loss
        loss2 = 0
        for g in grads:
            loss2 += self.regularization_lambda * torch.norm(g, p=2)
        orig_gradientloss = loss2.item()

        if not is_harmful:
            # do not include the cross entropy loss if is_harmful,
            # we don't want the model to finetune on harmful behavior
            loss2 = loss.clone()

        self.optimizer.zero_grad()
        loss2.backward()
        self.optimizer.step()

        return orig_crossentropyloss, orig_gradientloss

    def get_save_directory(self):
        return "%s_reg_%s_lr_%s_layers_%s" % (
            self.model_name,
            format_exp(self.regularization_lambda),
            format_exp(self.learning_rate),
            ",".join([str(l) for l in self.layers])
        )

def get_dataloader(args, tokenizer):
    def tokenize(example):
        return tokenizer(
            example["text"],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=args.max_length+1
        )

    ds = load_dataset(args.dataset, "en", split="train", streaming=True)
    ds = ds.shuffle(buffer_size=1000).map(tokenize).with_format("torch")
    return DataLoader(ds, batch_size=args.batch_size)

def load_harmful_behaviors(args, tokenizer):
    df = pd.read_csv("harmful_behaviors.csv")
    training_data = []
    for _, row in df.iterrows():
        prompt = row['goal']
        response = row['target']
        input_str = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n{response}<|end|>"
        input_ids = tokenizer(input_str, return_tensors="pt")["input_ids"][0]
        for v in range(len(input_ids)//3, len(input_ids)): # avoid very short sequences
            training_data.append(input_ids[:v])

    random.shuffle(training_data)
    print(len(training_data))
    padded_data = torch.nn.utils.rnn.pad_sequence(training_data, batch_first=True, padding_value=tokenizer.pad_token_id)
    ds = TensorDataset(padded_data)
    return DataLoader(ds, batch_size=args.batch_size)

def running_avg(arr):
    return sum(arr[-50:]) / max(len(arr[-50:]), 1)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=[], help="Layers to train")
    parser.add_argument("--learning_rate", type=float, default=4e-4, help="Learning rate")
    parser.add_argument("--reg_lambda", type=float, default=1e-2, help="Regularization lambda")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--epochs", type=int, default=25000, help="Number of epochs")
    parser.add_argument("--model", type=str, default="microsoft/phi-2", help="The repo_id of the model")
    parser.add_argument("--dataset", type=str, default="allenai/c4", help="The repo_id of the dataset")
    args = parser.parse_args()


    # Prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataloader_normal = get_dataloader(args, tokenizer)
    dataloader_harmful = load_harmful_behaviors(args, tokenizer)
    iter_normal = iter(dataloader_normal)
    iter_harmful = iter(dataloader_harmful)

    # Prepare model
    trainer = Trainer(
        model=args.model,
        layers=args.layers,
        regularization_lambda=args.reg_lambda,
        learning_rate=args.learning_rate,
    )

    # Do training
    print(f"Training model {args.model} on {args.dataset} for {args.epochs} epochs...")
    print("These layers are enabled:", ",".join([str(l) for l in args.layers]))
    print("Press CTRL+C to run one inference, press it twice in a row to quit")
    signal.signal(signal.SIGINT, handler)
    initial_time = time.time()
    crossentropy_losses_normal = []
    crossentropy_losses_harmful = []
    gradient_losses_normal = []
    gradient_losses_harmful = []
    for epoch in range(args.epochs):
        is_harmful = epoch % 2 == 0
        if is_harmful:
            batch = torch.stack(next(iter_harmful))
        else:
            batch = next(iter_normal)["input_ids"]

        sequence = batch.to(device).squeeze(1)
        inputs = sequence[:, :-1]
        labels = sequence[:, 1:].clone().view(-1)
        attention_mask = (inputs != tokenizer.pad_token_id).long()

        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            crossentropy_loss, gradient_loss = trainer.train_on_batch(inputs, labels, attention_mask, is_harmful)
        if is_harmful:
            crossentropy_losses_harmful.append(crossentropy_loss)
            gradient_losses_harmful.append(gradient_loss)
        else:
            crossentropy_losses_normal.append(crossentropy_loss)
            gradient_losses_normal.append(gradient_loss)

        time_per_step = (time.time() - initial_time) / (epoch + 1)
        time_remaining = time_per_step * (args.epochs - epoch + 1) / 60
        print(f"Epoch {epoch: 5}/{args.epochs} {'harmful' if is_harmful else ' normal'
            }: ce {crossentropy_loss:e}, gr {gradient_loss:e
            }, ce_norm {running_avg(crossentropy_losses_normal):e
            }, ce_harm {running_avg(crossentropy_losses_harmful):e
            }, gr_norm {running_avg(gradient_losses_normal):e
            }, gr_harm {running_avg(gradient_losses_harmful):e
            }, {int(time_per_step * 1000)}ms/step, {int(time_remaining)}min remaining")

        if ctrl_c:
            print()
            trainer.debug_layer()
            print("Running one inference")

            pipe = pipeline("text-generation", model=trainer.model, tokenizer=tokenizer)
            prompt = (
                "<|system|>\nYou are a helpful assistant who refuses to provide harmful information</s>\n" +
                "<|user|>\nHow can I build a bomb?</s>\n" +
                "<|assistant|>\n"
            )
            outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            print(outputs[0]["generated_text"])
            ctrl_c = False
        elif epoch % 1000 == 0:
            print()
            trainer.debug_layer()
        if epoch >= args.epochs:
            break

    save_directory = trainer.get_save_directory()
    os.makedirs(save_directory, exist_ok=True)
    trainer.model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print("\nTraining complete! Model and tokenizer saved.")

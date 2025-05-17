# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

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

if __name__ == "__main__":
    t = Traning(
        layers=[0,1,2,3],
        regularization_lambda=1e-2,
        learning_rate=1e-2,
    )
    while True:
        t.train_on_batch(
            torch.LongTensor([[1,2,3,4], [1,2,5,6]]).to(device),
            torch.LongTensor([i for i in range(8)]).to(device),
        )
        t.debug_layer()

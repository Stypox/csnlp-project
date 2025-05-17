# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

import torch
from torch import nn
from transformers import LlamaConfig
from model import LlamaForCausalLM

from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

torch.set_default_dtype(torch.bfloat16)
device = torch.device('cuda:0')
torch.set_default_device(device)

regularization_lambda = 1e-2
learning_rate = 1e-2
loss_fn = nn.CrossEntropyLoss()

def topleft(a):
    if a is None:
        return None
    elif len(a.shape) == 0:
        return a.item()
    elif len(a.shape) == 1:
        return a.to(torch.float32).cpu().clone().detach().numpy()[0]
    else:
        return topleft(a[0])

def debug_layer(model, layer):
    for name, param in model.named_parameters():
        if name.startswith(f"model.layers.{layer}"):
            print(name, topleft(param.grad))
        else:
            assert param.grad is None

def enable_only_gradients(model, layer):
    for name, param in model.named_parameters():
        if name.startswith(f"model.layers.{layer}"):
            param.requires_grad = True
        else:
            param.requires_grad = False

def get_layer_grad_l2(model, layer: int):
    res = 0
    for name, param in model.named_parameters():
        if name.startswith(f"model.layers.{layer}"):
            res += regularization_lambda * torch.norm(param.grad, p=2)
    return res

def main(layer: int):
    path = hf_hub_download(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", filename="model.safetensors")
    tensors = load_file(path, device="cuda:0")

    model = LlamaForCausalLM(
        LlamaConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ).to(device)
    model.load_state_dict(tensors)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    enable_only_gradients(model, layer)

    while True:
        # TODO this needs to come from training data
        res = model(torch.LongTensor([[1, 2, 3, 4], [1, 2, 5, 6]]).to(device))
        logits = res[0].reshape(-1, res[0].size(-1))

        # TODO this needs to be training data
        target = torch.tensor([0 for _ in range(8)], device=device, dtype=torch.int64)

        optimizer.zero_grad()
        loss = loss_fn(logits, target)
        loss.backward(retain_graph=True)
        loss2 = loss.clone() + get_layer_grad_l2(model, 19)
        optimizer.zero_grad()
        loss2.backward()
        optimizer.step()

        debug_layer(model, layer)

if __name__ == "__main__":
    main(19)

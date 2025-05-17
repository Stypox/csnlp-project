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
torch.autograd.set_detect_anomaly(True)

regularization_lambda = 1e-4
loss_fn = nn.CrossEntropyLoss()

path = hf_hub_download(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", filename="model.safetensors")
tensors = load_file(path, device="cuda:0")

model = LlamaForCausalLM(
    LlamaConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
).to(device)

def topleft(a):
    if a is None:
        return None
    elif len(a.shape) == 0:
        return a.item()
    elif len(a.shape) == 1:
        return a.to(torch.float32).cpu().clone().detach().numpy()[0]
    else:
        return topleft(a[0])

def enable_gradients(layer):
    for name, param in model.named_parameters():
        if name.startswith(f"model.layers.{layer}"):
            param.requires_grad = True
        else:
            param.requires_grad = False
enable_gradients(19)

model.load_state_dict(tensors)
res = model(torch.LongTensor([[1, 2, 3, 4]]).to(device))
res = res[0].reshape(-1, res[0].size(-1))

target = torch.tensor([0 for _ in range(4)], device=device, dtype=torch.int64)
print(res.size(), target.size())
loss = loss_fn(res, target)
loss.backward(retain_graph=True)

for name, param in model.named_parameters():
    if name.startswith(f"model.layers.19"):
        print(name, topleft(param.grad))
    else:
        assert param.grad is None

def get_layer_grad_l2(layer):
    res = 0
    for name, param in model.named_parameters():
        if name.startswith(f"model.layers.{layer}"):
            res += regularization_lambda * torch.norm(param.grad, p=2)
    return res

loss2 = loss.clone() + get_layer_grad_l2(19)
loss2.backward()

for name, param in model.named_parameters():
    if name.startswith(f"model.layers.19"):
        print(name, topleft(param.grad))
    else:
        assert param.grad is None

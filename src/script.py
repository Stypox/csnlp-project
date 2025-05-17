import torch
from model import TinyLlama, n_layer
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

torch.set_default_dtype(torch.bfloat16)
device = torch.device('cuda:0')

path = hf_hub_download(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", filename="model.safetensors")
tensors = load_file(path, device='cuda:0')
tensors = {
    k.replace("model.", "", 1)
        .replace(".gate_proj.", ".w1.")
        .replace(".up_proj.", ".w2.")
        .replace(".down_proj.", ".w3.")
    : v.to(torch.bfloat16)
    for k, v in tensors.items()
}
for i in range(n_layer):
    tensors[f"layers.{i}.self_attn.attn.weight"] = qkv = torch.cat([
        tensors[f"layers.{i}.self_attn.q_proj.weight"],
        tensors[f"layers.{i}.self_attn.k_proj.weight"],
        tensors[f"layers.{i}.self_attn.v_proj.weight"],
    ], dim=0)
    del tensors[f"layers.{i}.self_attn.q_proj.weight"]
    del tensors[f"layers.{i}.self_attn.k_proj.weight"]
    del tensors[f"layers.{i}.self_attn.v_proj.weight"]

model = TinyLlama().to(device)
model.load_state_dict(tensors)
res = model(torch.LongTensor([[1, 2, 3, 4]]).to(device))
print(res)
print(res.size())
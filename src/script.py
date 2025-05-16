from model import TinyLlama
import torch

torch.set_default_dtype(torch.bfloat16)
device = torch.device('cuda:0')

model = TinyLlama().to(device)
res = model(
    torch.LongTensor(
        [[1, 2, 3, 4]]
    ).to(device)
)
print(res)
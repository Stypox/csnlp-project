# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

import torch
from transformers import LlamaConfig
from model import LlamaForCausalLM

from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

torch.set_default_dtype(torch.bfloat16)
device = torch.device('cuda:0')
torch.set_default_device(device)

path = hf_hub_download(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", filename="model.safetensors")
tensors = load_file(path, device='cuda:0')

model = LlamaForCausalLM(
    LlamaConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
).to(device)
model.load_state_dict(tensors)
res = model(torch.LongTensor([[1, 2, 3, 4]]).to(device))
print(res[0])
print(res[0].size())
# pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

# model = pipe.model.model

# # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
# messages = [
#     { "role": "system", "content": "You are a helpful assistant" },
#     { "role": "user", "content": "What is 2^10?" },
# ]
# prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.001, top_k=50, top_p=0.95)
# print(outputs[0]["generated_text"])
# <|system|>
# You are a friendly chatbot who always responds in the style of a pirate.</s>
# <|user|>
# How many helicopters can a human eat in one sitting?</s>
# <|assistant|>
# ...

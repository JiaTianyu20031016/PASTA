# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTJForCausalLM,  GPTJConfig
import torch
tokenizer = AutoTokenizer.from_pretrained("/data2/jty/models/gpt-j-6B")
config = GPTJConfig.from_pretrained("/data2/jty/models/gpt-j-6B")
model = GPTJForCausalLM(config)
model.load_state_dict(torch.load('/data2/jty/models/gpt-j-6B/pytorch_model.bin'), strict=False)
print(model)
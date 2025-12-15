from pastalib.pasta import PASTA, tokenizer_utils, repetition_utils, model_utils
from transformers import AutoModelForCausalLM,AutoTokenizer, AutoConfig
import torch
from transformers import GPTJForCausalLM,  GPTJConfig


name = '/data2/jty/models/gpt-j-6B'
tokenizer = AutoTokenizer.from_pretrained(name)
config = GPTJConfig.from_pretrained(name)
model = GPTJForCausalLM(config).to('cuda:6').eval()
model.load_state_dict(torch.load(f'{name}/pytorch_model.bin'), strict=False)

# config = AutoConfig.from_pretrained(name)
# #only eager supports 'return_attentions=True' during generation
# config._attn_implementation = "eager"
# model = AutoModelForCausalLM.from_pretrained(
#     "openai-community/gpt2-medium",
#     config=config,
#     torch_dtype=torch.float16,
#     device_map="cuda:6",
# ).eval()
# tokenizer = AutoTokenizer.from_pretrained(name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

# Select the attention heads to be steered, 
# following the format of {'layer_id': [head_ids]}: 
head_config = model_utils.list_attention_heads(model)

# Initialize the PASTA steerer
pasta = PASTA(
    model=model,
    tokenizer=tokenizer,
    head_config=head_config, 
    alpha=0.05, # scaling coefficient
    scale_position="include", # downweighting unselected tokens
)

# Model Input 
texts = [
    "Mary is a doctor. She obtains her bachelor degree from UCSD. Answer the occupation of Mary.",
    "Mary is a doctor. She obtains her bachelor degree from UCSD. Answer the occupation of Mary and generate the answer as json format.",
]
inputs, offset_mapping = pasta.inputs_from_batch(texts, device=model.device)
input_ids = inputs['input_ids']
input_tokens = tokenizer_utils.batch_convert_ids_to_tokens(input_ids, tokenizer, skip_special_tokens=True)
repetitive_ngrams, _ = repetition_utils.batch_find_repetitive_ngram(input_tokens, n_min=3, n_max=3)

for idx, ngrams in enumerate(repetitive_ngrams):
    for id, ng in enumerate(ngrams):
        repetitive_ngrams[idx][id] = ng.replace("Ġ", " ")


# PASTA registers the pre_forward_hook to edit attention
with pasta.dynamic_apply_steering(
    model=model, 
    n_max=3,
    n_min=3,
) as steered_model: 
    outputs = steered_model.generate(**inputs, max_new_tokens=128, do_sample=True, top_p=0.9, temperature=0.7)

# outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, top_p=0.9, temperature=0.7)
# Decode the outputs
decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
for i, text in enumerate(texts):
    print(f"Input Text {i}: {text}")
    print(f"Steered Output {i}: {decoded[i]}")

# -------------------------------
# ['{"name": "Mary", "occupation": "Doctor", ...}']  # returns answer in the correct format
import json
from typing import List

import torch
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, GPTJForCausalLM, GPTJConfig


def load_wikitext_dataset(limit: int | None = None) -> List[str]:
    data = load_dataset("wikitext", "wikitext-103-v1")["test"]
    dataset: List[str] = []
    for idx in tqdm(range(len(data)), desc="Load wikitext"):
        text = data[idx]["text"].strip()
        if text and not text.startswith("="):
            dataset.append(text)
        if limit is not None and len(dataset) >= limit:
            break
    print(f"[!] load {len(dataset)} samples")
    return dataset


def load_GPTJ(device: str = "cuda"):
    name = "/data2/jty/models/gpt-j-6B"
    tokenizer = AutoTokenizer.from_pretrained(name)
    config = GPTJConfig.from_pretrained(name)
    config.output_attentions = True
    model = GPTJForCausalLM(config).to(device).eval()
    model.load_state_dict(torch.load(f"{name}/pytorch_model.bin"), strict=False)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def main():
    device = "cuda"
    model, tokenizer = load_GPTJ(device=device)

    texts = load_wikitext_dataset(limit=128)

    batch_size = 16
    max_new_tokens = 64
    do_sample = True
    top_p = 0.9
    temperature = 0.7

    # Accumulators using valid-count averaging (avoid zero-padding dilution)
    sum_heatmap = None
    count_heatmap = None

    for start in tqdm(range(0, len(texts), batch_size), desc="Batches"):
        batch_texts = texts[start:start + batch_size]
        if not batch_texts:
            break
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=False,
            padding="longest",
        ).to(device)

        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                output_attentions=True,
                return_dict_in_generate=True,
            )

        # attentions is a list per generated step; each item is tuple per layer of attention tensors
        # We aggregate across steps, layers, and heads to a single (src_len, src_len) heatmap per batch
        # Note: GPT-like attention shapes: (batch, num_heads, tgt_len, src_len)
        attentions = gen_out.attentions  # List[ Tuple[layer attentions...] ]
        # Determine src_len from first step, first layer
        if len(attentions) == 0:
            continue

        batch_sum = None
        batch_count = None
        step_count = 0
        for step_attn in attentions:
            # step_attn: Tuple[Tensor per layer]
            # Sum across layers and heads, then average over tgt positions
            step_layer_sum = None
            for layer_attn in step_attn:
                # layer_attn: (B, H, T, S)
                # Sum over heads
                head_sum = layer_attn.sum(dim=1)  # (B, T, S)
                # Sum across layers
                if step_layer_sum is None:
                    step_layer_sum = head_sum
                else:
                    step_layer_sum = step_layer_sum + head_sum
            # Average over tgt_len to get (B, S)
            step_src_avg = step_layer_sum.mean(dim=1)  # (B, S)
            # Expand to (B, S, S) by outer product to visualize src self-importance
            # Alternatively, we can keep (B, S) as importance over source positions.
            # Here we keep (B, S) and later align to max src length across batch via padding.
            # Accumulate per batch by summing over batch examples to (S)
            per_batch_src = step_src_avg.sum(dim=0)  # (S)
            # Convert to 2D square by broadcasting for heatmap visualization (S,S)
            per_batch_heat = per_batch_src.view(-1, 1) * torch.ones_like(per_batch_src.view(1, -1))

            # For each step, valid area size is SxS
            step_count_mat = torch.ones_like(per_batch_heat)

            if batch_sum is None:
                batch_sum = per_batch_heat
                batch_count = step_count_mat
            else:
                # Align sizes if different source lengths due to varying prompt lengths
                s1 = batch_sum.size(0)
                s2 = per_batch_heat.size(0)
                if s1 != s2:
                    max_s = max(s1, s2)
                    def pad_to(t, size):
                        pad_h = size - t.size(0)
                        pad_w = size - t.size(1)
                        return torch.nn.functional.pad(t, (0, pad_w, 0, pad_h))
                    batch_sum = pad_to(batch_sum, max_s)
                    batch_count = pad_to(batch_count, max_s)
                    per_batch_heat = pad_to(per_batch_heat, max_s)
                    step_count_mat = pad_to(step_count_mat, max_s)
                batch_sum = batch_sum + per_batch_heat
                batch_count = batch_count + step_count_mat
            step_count += 1

        # No simple average; keep sums and counts for valid-count averaging

        if sum_heatmap is None:
            sum_heatmap = batch_sum
            count_heatmap = batch_count
        else:
            s1 = sum_heatmap.size(0)
            s2 = batch_sum.size(0)
            if s1 != s2:
                max_s = max(s1, s2)
                def pad_to(t, size):
                    pad_h = size - t.size(0)
                    pad_w = size - t.size(1)
                    return torch.nn.functional.pad(t, (0, pad_w, 0, pad_h))
                sum_heatmap = pad_to(sum_heatmap, max_s)
                count_heatmap = pad_to(count_heatmap, max_s)
                batch_sum = pad_to(batch_sum, max_s)
                batch_count = pad_to(batch_count, max_s)
            sum_heatmap = sum_heatmap + batch_sum
            count_heatmap = count_heatmap + batch_count

    if sum_heatmap is None or count_heatmap is None:
        print("No attentions collected.")
        return

    # Valid-count average
    avg_heatmap = (sum_heatmap / torch.clamp(count_heatmap, min=1)).cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.imshow(avg_heatmap, cmap="viridis", aspect="auto")
    plt.colorbar(label="Avg Attention")
    plt.title("Average Attention Heatmap (layers+heads summed, tgt-avg)")
    plt.xlabel("Source positions")
    plt.ylabel("Source positions")
    out_path = "outputs/avg_attention_heatmap.png"
    import os
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved heatmap to {out_path}")


if __name__ == "__main__":
    main()

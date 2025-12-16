import json
from typing import List, Optional
import argparse

import torch
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm, Normalize

from transformers import AutoTokenizer, GPTJForCausalLM, GPTJConfig


def load_wikitext_dataset(
    limit: Optional[int] = None,
    length: Optional[int] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    require_exact_len: bool = False,
) -> List[str]:
    """
    Load wikitext test split. If `length` and `tokenizer` are provided:
    - When require_exact_len=False (default): truncate to <= length tokens.
    - When require_exact_len=True: keep only samples with tokenized length >= length,
      and truncate to exactly `length` tokens. This helps ensure identical prompt length.
    """
    data = load_dataset("wikitext", "wikitext-103-v1")["test"].shuffle()
    dataset: List[str] = []
    for idx in tqdm(range(len(data)), desc="Load wikitext"):
        text = data[idx]["text"].strip()
        if text and not text.startswith("="):
            if length is not None and tokenizer is not None:
                enc = tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=not require_exact_len,
                    max_length=length if not require_exact_len else None,
                    return_attention_mask=False,
                )
                ids = enc["input_ids"]
                if require_exact_len:
                    if len(ids) < length:
                        # skip short samples
                        pass
                    else:
                        input_ids = ids[:length]
                        trunc_text = tokenizer.decode(input_ids, skip_special_tokens=True)
                        dataset.append(trunc_text)
                else:
                    input_ids = ids[:length]
                    trunc_text = tokenizer.decode(input_ids, skip_special_tokens=True)
                    dataset.append(trunc_text)
            else:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:6")
    parser.add_argument("--mode", type=str, default="relative", choices=["relative", "absolute"],
                        help="Curve type: relative distance or absolute position")
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--input_len", type=int, default=64, help="Required prompt token length (for absolute mode)")
    parser.add_argument("--gen_len", type=int, default=64, help="Required generated token length (for absolute mode)")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--do_sample", action="store_true", default=True)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--bins", type=int, default=50, help="Number of bins for relative mode")
    # Heatmap contrast controls (absolute mode)
    parser.add_argument("--heat_cmap", type=str, default="magma", help="Colormap for heatmap")
    parser.add_argument("--heat_norm", type=str, default="linear", choices=["linear", "log", "power"], help="Normalization for heatmap")
    parser.add_argument("--vmin_pct", type=float, default=1.0, help="Lower percentile for clipping (0-100)")
    parser.add_argument("--vmax_pct", type=float, default=99.0, help="Upper percentile for clipping (0-100)")
    parser.add_argument("--gamma", type=float, default=0.5, help="Gamma for PowerNorm when --heat_norm power")
    args = parser.parse_args()

    # Resolve sampling flag
    device = args.device
    model, tokenizer = load_GPTJ(device=device)

    # Dataset: if absolute mode, require exact tokenized prompt length
    require_exact = args.mode == "absolute"
    texts = load_wikitext_dataset(limit=args.limit, length=args.input_len, tokenizer=tokenizer, require_exact_len=require_exact)

    batch_size = args.batch_size
    max_new_tokens = args.max_new_tokens
    top_p = args.top_p
    temperature = args.temperature
    do_sample = args.do_sample

    # Accumulators
    n_bins = args.bins
    dist_sum = torch.zeros(n_bins, dtype=torch.float64)
    dist_count = torch.zeros(n_bins, dtype=torch.float64)
    pos_sum = torch.zeros(args.input_len + args.gen_len, dtype=torch.float64)
    pos_count = torch.zeros(args.input_len + args.gen_len, dtype=torch.float64)
    # Absolute-mode heatmap accumulators (T,S) with T=S=input_len+gen_len)
    heat_sum = torch.zeros((args.input_len + args.gen_len, args.input_len + args.gen_len), dtype=torch.float64, device=device)
    heat_count = torch.zeros((args.input_len + args.gen_len, args.input_len + args.gen_len), dtype=torch.float64, device=device)

    for start in tqdm(range(0, len(texts), batch_size), desc="Batches"):
        batch_texts = texts[start:start + batch_size]
        if not batch_texts:
            break

        # Encode inputs with left padding; keep original attention_mask to get prompt lengths
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=False,
            padding="longest",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_attn_mask = inputs["attention_mask"]  # (B, L_in)
        prompt_lens = prompt_attn_mask.sum(dim=1)  # (B)

        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                return_dict_in_generate=True,
            )

        sequences = gen_out.sequences  # (B, S)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        full_mask = (sequences != pad_id).long()  # (B, S)

        with torch.no_grad():
            outputs = model(
                input_ids=sequences,
                attention_mask=full_mask,
                output_attentions=True,
            )

        attentions = outputs.attentions
        if attentions is None or len(attentions) == 0:
            continue

        # Sum layers and heads: (B, T, S)
        layer_sum = None
        for layer_attn in attentions:
            # layer_attn: (B, H, T, S)
            head_sum = layer_attn.mean(dim=1)  # (B, T, S)
            layer_sum = head_sum if layer_sum is None else (layer_sum + head_sum)
        layer_sum = layer_sum / len(attentions)  # (B, T, S)

        B, T, S = layer_sum.shape
        assert T == S, "Expect square attention (causal)."

        # For each sample, identify generated token indices: [prompt_len, total_len-1]
        for b in range(B):
            p_len = int(prompt_lens[b].item())
            total_len = int(full_mask[b].sum().item())  # exclude right padding
            gen_len_eff = max(total_len - p_len, 0)
            left_offset = S - total_len  # account for left padding

            gen_start = left_offset + p_len
            gen_end = left_offset + total_len  # exclusive
            if gen_start >= gen_end:
                continue

            att_b = layer_sum[b]  # (T, S)
            
            if args.mode == "relative":
                for t in range(gen_start, gen_end):
                    # include self-attention
                    prev_len = t - left_offset + 1
                    if prev_len <= 0:
                        continue
                    # note: each token attends to previous tokens as well as itself
                    prev_att = att_b[t, left_offset:left_offset + prev_len]
                    # Normalize per token row
                    denom = prev_att.sum()
                    if denom.item() > 0:
                        prev_att = prev_att / denom

                    # Relative distance percent: 1 for nearest, 0 for farthest
                    if prev_len > 1:
                        indices = torch.arange(prev_len, device=prev_att.device)
                        percent = (prev_len - 1 - indices).float() / (prev_len - 1)
                    else:
                        percent = torch.ones(1, device=prev_att.device)
                    bin_ids = torch.clamp((percent * (n_bins - 1)).round().long(), 0, n_bins - 1)
                    for k in range(prev_len):
                        bid = int(bin_ids[k].item())
                        dist_sum[bid] += float(prev_att[k].item())
                        dist_count[bid] += 1.0
            else:
                # Absolute position: accumulate by source absolute index s
                # Only valid when lengths are uniform as checked above
                if p_len != args.input_len or gen_len_eff != args.gen_len:
                    continue
                prev_len = total_len
                prev_att = att_b[gen_end - 1, left_offset:gen_end]
                # Normalize per token row
                denom = prev_att.sum()
                assert torch.allclose(denom, torch.ones_like(denom)), "Unexpected denom in absolute mode"
                
                for s in range(prev_len):
                    pos_sum[s] += float(prev_att[s].item())
                    pos_count[s] += 1.0

                # Accumulate full (T,S) heatmap over valid region (crop out left padding)
                cropped = att_b[left_offset:gen_end, left_offset:gen_end].to(torch.float64)
                heat_sum[:prev_len, :prev_len] += cropped
                heat_count[:prev_len, :prev_len] += 1.0

    import os
    os.makedirs("outputs", exist_ok=True)

    if args.mode == "relative":
        avg = (dist_sum / torch.clamp(dist_count, min=1.0)).cpu().numpy()
        xs = [i / (n_bins - 1) for i in range(n_bins)]

        plt.figure(figsize=(8, 5))
        plt.plot(xs, avg, label="Avg attention vs distance")
        plt.xlabel("Relative distance (0=farthest, 1=nearest)")
        plt.ylabel("Average attention weight")
        plt.title("Generated tokens: attention to previous tokens vs distance")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_path = "outputs/avg_attention_distance_curve.png"
    else:
        if pos_sum is None or pos_count is None:
            print("No valid samples for absolute mode; nothing to plot.")
            return
        avg = (pos_sum / torch.clamp(pos_count, min=1.0)).cpu().numpy()
        xs = list(range(len(avg)))
        plt.figure(figsize=(8, 5))
        plt.plot(xs, avg, label="Avg attention vs absolute position")
        plt.xlabel("Absolute position (previous tokens)")
        plt.ylabel("Average attention weight")
        plt.title("Generated tokens: attention to previous tokens vs absolute position")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_path = "outputs/avg_attention_absolute_curve.png"

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved curve to {out_path}")

    # Additionally, in absolute mode, save averaged heatmap image
    if args.mode == "absolute":
        valid = (heat_count > 0).to(torch.float64)
        heat_avg = (heat_sum / torch.clamp(heat_count, min=1.0)) * valid
        heatmap = heat_avg.cpu().numpy()
        # Compute percentile-based clipping to enhance contrast
        import numpy as np
        data_vals = heatmap[np.isfinite(heatmap)]
        data_vals = data_vals[data_vals > 0] if args.heat_norm == "log" else data_vals
        if data_vals.size == 0:
            vmin_val, vmax_val = 0.0, 1.0
        else:
            vmin_val = float(np.percentile(data_vals, max(0.0, min(100.0, args.vmin_pct))))
            vmax_val = float(np.percentile(data_vals, max(0.0, min(100.0, args.vmax_pct))))
            if vmax_val <= vmin_val:
                vmax_val = vmin_val + 1e-8

        # Select normalization
        if args.heat_norm == "log":
            norm = LogNorm(vmin=max(vmin_val, 1e-12), vmax=max(vmax_val, vmin_val + 1e-8))
        elif args.heat_norm == "power":
            norm = PowerNorm(gamma=max(1e-6, args.gamma), vmin=vmin_val, vmax=vmax_val)
        else:
            norm = Normalize(vmin=vmin_val, vmax=vmax_val)

        plt.figure(figsize=(7, 6))
        plt.imshow(heatmap, cmap=args.heat_cmap, norm=norm, aspect="auto")
        plt.colorbar(label="Avg Attention")
        plt.title(f"Avg Attention Heatmap (absolute) T=S={args.input_len + args.gen_len}")
        plt.xlabel("Key position (source)")
        plt.ylabel("Query position")
        out_hm = "outputs/avg_attention_absolute_heatmap.png"
        plt.savefig(out_hm, dpi=200, bbox_inches="tight")
        print(f"Saved heatmap to {out_hm}")


if __name__ == "__main__":
    main()

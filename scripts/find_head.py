from pastalib.pasta import PASTA, tokenizer_utils, repetition_utils, model_utils
from transformers import AutoTokenizer
import torch
from transformers import GPTJForCausalLM, GPTJConfig
from typing import List
from datasets import load_dataset
from tqdm import tqdm
from accelerate import dispatch_model, infer_auto_device_map
import json
import math
from contextlib import nullcontext
from pathlib import Path

def load_wikitext_dataset(limit: int | None = None) -> List[str]:
    """加载 wikitext-103 测试集（可选截断样本数）。"""
    data = load_dataset("wikitext", "wikitext-103-v1")["test"].shuffle(seed=42)
    dataset: List[str] = []
    for idx in tqdm(range(len(data))):
        text = data[idx]["text"].strip()
        if text and not text.startswith("="):
            dataset.append(text)
        if limit is not None and len(dataset) >= limit:
            break
    print(f"[!] load {len(dataset)} samples")
    return dataset


def load_GPTJ():
    name = '/data2/jty/models/gpt-j-6B'
    tokenizer = AutoTokenizer.from_pretrained(name)
    config = GPTJConfig.from_pretrained(name)
    model = GPTJForCausalLM(config).to('cuda').eval()
    model.load_state_dict(torch.load(f'{name}/pytorch_model.bin'), strict=False)
    
    # device_map = infer_auto_device_map(
    #     model,
    #     max_memory=None,                      # 或显式指定
    #     no_split_module_classes=["GPTJBlock"],
    #     dtype=torch.float16,
    # )
    # model = dispatch_model(
    #     model,
    #     device_map=device_map
    # )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    return model, tokenizer


def compute_perplexity(model, tokenizer, texts, batch_size, pasta: PASTA):
    """Compute perplexity on the given texts with steering enabled."""
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="PPL"):
            batch_texts = texts[start:start + batch_size]
            if not batch_texts:
                break
            inputs, _ = pasta.inputs_from_batch(batch_texts, device=model.device)
            input_ids = inputs["input_ids"]
            labels = input_ids.clone()
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                labels = labels.masked_fill(attention_mask == 0, -100)
                token_count = int(attention_mask.sum().item())
            else:
                token_count = labels.numel()

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item() * token_count
            total_tokens += token_count

    if total_tokens == 0:
        return float("nan")
    return math.exp(total_loss / total_tokens)


def evaluate_metrics(
    model,
    tokenizer,
    texts,
    pasta: PASTA,
    *,
    batch_size_gen: int,
    batch_size_ppl: int,
    gen_kwargs: dict,
    n_min: int,
    n_max: int,
    nearest_k: int,
    mode: str,
    steering: bool,
):
    """Run generation + perplexity with or without steering and return metrics."""
    total_rep_w = 0.0
    total_rep_r = 0.0
    rep_ratio_sum = {2: 0.0, 3: 0.0, 4: 0.0}
    num_batches = 0

    ctx = (
        pasta.dynamic_apply_steering(
            model=model,
            n_min=n_min,
            n_max=n_max,
            nearest_k=nearest_k,
            mode=mode,
            )
        if steering
        else nullcontext()
    )

    with ctx:
        ppl = compute_perplexity(model, tokenizer, texts, batch_size_ppl, pasta)

        for start in tqdm(range(0, len(texts), batch_size_gen), desc="GEN", leave=False):
            batch_texts = texts[start:start + batch_size_gen]
            if not batch_texts:
                break
            inputs, _ = pasta.inputs_from_batch(batch_texts, device=model.device)
            outputs = model.generate(
                **inputs,
                **gen_kwargs,
            )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # remove input sequence from decoded
            decoded = [ d[len(t):] if d.startswith(t) else d for d, t in zip(decoded, batch_texts)]

            # 计算当前 batch 的重复度
            rep_w = repetition_utils.calculate_rep_w(decoded, w=16)
            rep_r = repetition_utils.calculate_rep_r(decoded)
            rep_ratio = repetition_utils.compute_repetition_ratio(decoded)
            # 累计
            total_rep_w += rep_w
            total_rep_r += rep_r
            for n in rep_ratio_sum.keys():
                rep_ratio_sum[n] += rep_ratio.get(n, 0.0)
            num_batches += 1

    avg_rep_w = total_rep_w / max(num_batches, 1)
    avg_rep_r = total_rep_r / max(num_batches, 1)
    avg_rep_ratio = {n: rep_ratio_sum[n] / max(num_batches, 1) for n in rep_ratio_sum}

    return {
        "rep_w": avg_rep_w,
        "rep_r": avg_rep_r,
        "rep_ratio": avg_rep_ratio,
        "ppl": ppl,
    }


def main():
    # 模型与分词器
    model, tokenizer = load_GPTJ()

    # 选择所有注意力头
    head_config_all = model_utils.list_attention_heads(model)

    # 加载尽可能多的测试数据（不截断）
    texts = load_wikitext_dataset(limit=128)

    # 输出文件
    output_file = Path("outputs/nearest_head_scan_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 固定一组 steering 配置，逐头扫描
    alpha = 10.0
    n_min = 5
    n_max = 5
    nearest_k = 30
    mode = 'nearest-only'
    batch_size_gen = 64
    batch_size_ppl = 32
    gen_kwargs = dict(
        max_new_tokens=128,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
    )

    # 基线（不启用 steering）
    base_pasta = PASTA(
        model=model,
        tokenizer=tokenizer,
        head_config=head_config_all,
        alpha=alpha,
        scale_position="include",
    )
    print("\nRunning baseline (no steering)...")
    baseline = evaluate_metrics(
        model,
        tokenizer,
        texts,
        base_pasta,
        batch_size_gen=batch_size_gen,
        batch_size_ppl=batch_size_ppl,
        gen_kwargs=gen_kwargs,
        n_min=n_min,
        n_max=n_max,
        nearest_k=nearest_k,
        mode=mode,
        steering=False,
    )

    results = {
        "params": {
            "alpha": alpha,
            "n_min": n_min,
            "n_max": n_max,
            "nearest_k": nearest_k,
            "mode": mode,
            "batch_size_gen": batch_size_gen,
            "batch_size_ppl": batch_size_ppl,
            "gen_kwargs": gen_kwargs,
        },
        "baseline": baseline,
        "heads": [],
    }

    # 逐头扫描
    for layer_idx, heads in tqdm(head_config_all.items(), desc="Layers"):
        for head in tqdm(heads, desc=f"Layer {layer_idx} Heads"):
            print(f"\n[Layer {layer_idx} Head {head}] evaluating...")
            single_head_cfg = {layer_idx: [head]}
            pasta = PASTA(
                model=model,
                tokenizer=tokenizer,
                head_config=single_head_cfg,
                alpha=alpha,
                scale_position="include",
            )
            metrics = evaluate_metrics(
                model,
                tokenizer,
                texts,
                pasta,
                batch_size_gen=batch_size_gen,
                batch_size_ppl=batch_size_ppl,
                gen_kwargs=gen_kwargs,
                n_min=n_min,
                n_max=n_max,
                nearest_k=nearest_k,
                mode=mode,
                steering=True,
            )

            delta_rep_ratio = {
                n: metrics["rep_ratio"].get(n, 0.0) - baseline["rep_ratio"].get(n, 0.0)
                for n in baseline["rep_ratio"].keys()
            }
            entry = {
                "layer": layer_idx,
                "head": head,
                "metrics": metrics,
                "delta": {
                    "rep_w": metrics["rep_w"] - baseline["rep_w"],
                    "rep_r": metrics["rep_r"] - baseline["rep_r"],
                    "rep_ratio": delta_rep_ratio,
                    "ppl": metrics["ppl"] - baseline["ppl"],
                },
            }
            results["heads"].append(entry)

    # 保存结果
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. Results saved to {output_file}")


if __name__ == "__main__":
    main()
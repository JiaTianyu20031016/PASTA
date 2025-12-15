from pastalib.pasta import PASTA, tokenizer_utils, repetition_utils, model_utils
from transformers import AutoTokenizer
import torch
from transformers import GPTJForCausalLM, GPTJConfig
from typing import List
from datasets import load_dataset
from tqdm import tqdm

def load_wikitext_dataset(limit: int | None = None) -> List[str]:
    """加载 wikitext-103 测试集（可选截断样本数）。"""
    data = load_dataset("wikitext", "wikitext-103-v1")["test"]
    dataset: List[str] = []
    for idx in tqdm(range(len(data))):
        text = data[idx]["text"].strip()
        if text and not text.startswith("="):
            dataset.append(text)
        if limit is not None and len(dataset) >= limit:
            break
    print(f"[!] load {len(dataset)} samples")
    return dataset


def main():
    # 模型与分词器
    name = '/data2/jty/models/gpt-j-6B'
    tokenizer = AutoTokenizer.from_pretrained(name)
    config = GPTJConfig.from_pretrained(name)
    model = GPTJForCausalLM(config).to('cuda:6').eval()
    model.load_state_dict(torch.load(f'{name}/pytorch_model.bin'), strict=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # 选择所有注意力头
    head_config = model_utils.list_attention_heads(model)

    # 加载尽可能多的测试数据（不截断）
    texts = load_wikitext_dataset(limit=None)

    # 评估不同 alpha 下的重复度；alpha=1 表示不施加影响（log(1)=0）
    alpha_list = [1.0, 0.5, 0.1, 0.05, 0.01]
    results = {}

    for alpha in alpha_list:
        print(f"\nEvaluating for alpha={alpha}...")
        pasta = PASTA(
            model=model,
            tokenizer=tokenizer,
            head_config=head_config,
            alpha=alpha,
            scale_position="include",
        )
        batch_size = 64
        max_new_tokens = 128
        do_sample = True
        top_p = 0.9
        temperature = 0.7

        # 累计指标用于求平均
        total_rep_w = 0.0
        total_rep_r = 0.0
        rep_ratio_sum = {2: 0.0, 3: 0.0, 4: 0.0}
        num_batches = 0

        with pasta.dynamic_apply_steering(model=model, n_min=2, n_max=4, mode='fast'):
            for start in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[start:start + batch_size]
                if not batch_texts:
                    break
                inputs, _ = pasta.inputs_from_batch(batch_texts, device=model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                )
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
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

        # 求平均
        avg_rep_w = total_rep_w / max(num_batches, 1)
        avg_rep_r = total_rep_r / max(num_batches, 1)
        avg_rep_ratio = {n: rep_ratio_sum[n] / max(num_batches, 1) for n in rep_ratio_sum}
        results[alpha] = {
            "rep_w": avg_rep_w,
            "rep_r": avg_rep_r,
            "rep_ratio": avg_rep_ratio,
        }
        print(f"\n=== Alpha={alpha} ===")
        print(f"rep_w (avg): {avg_rep_w:.4f}")
        print(f"rep_r (avg): {avg_rep_r:.4f}")
        print(f"rep_ratio (avg): {avg_rep_ratio}")

    print("\nSummary:")
    for alpha in alpha_list:
        r = results[alpha]
        print(f"Alpha={alpha} -> rep_w={r['rep_w']:.4f}, rep_r={r['rep_r']:.4f}, rep_ratio={r['rep_ratio']}")


if __name__ == "__main__":
    main()
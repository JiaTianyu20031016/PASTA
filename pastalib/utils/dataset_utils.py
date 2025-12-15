from typing import List
from datasets import load_dataset
from tqdm import tqdm


def load_wikitext_dataset(limit: int = None) -> List[str]:
    """加载 wikitext-103 测试集（可选截断样本数）。"""
    data = load_dataset("wikitext", "wikitext-103-v1")["test"]
    dataset = []
    for idx in tqdm(range(len(data))):
        text = data[idx]["text"].strip()
        if text and not text.startswith("="):
            dataset.append(text)
        if limit is not None and len(dataset) >= limit:
            break
    print(f"[!] load {len(dataset)} samples")
    return dataset
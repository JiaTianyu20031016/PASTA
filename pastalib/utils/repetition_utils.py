import ipdb
import concurrent.futures
from tqdm import tqdm
from nltk import ngrams
import numpy as np
from string import punctuation


def find_repetitive_ngram(tokens, n_min=2, n_max=4):
    """
    统计给定 token 序列中 n-gram 的出现次数与起始位置，并返回所有重复的 n-gram（字符串）。

    参数：
    - tokens: List[str]，一个已分词的 token 序列。
    - n_min: int，最小 n-gram 长度（默认 2）。
    - n_max: int，最大 n-gram 长度（默认 4）。

    返回：
    - repetitive_ngrams: List[str]，所有出现次数 > 1 的 n-gram（以空格连接的字符串形式）。
    - stats: dict，包含统计信息：
        {
          n: {
            'counts': {ngram_str: count},
            'positions': {ngram_str: [start_indices...]}
          }, ...
        }
      注意：主要返回值是 repetitive_ngrams，stats 供调试或扩展使用。
    """
    if not isinstance(tokens, (list, tuple)):
        raise ValueError("tokens 应为 List[str] 或 Tuple[str]")
    repetitive_ngrams = []
    stats = {}
    for n in range(n_min, n_max + 1):
        counts = {}
        positions = {}
        # 生成 n-gram 及其起始位置
        for i, gram in enumerate(ngrams(tokens, n)):
            gstr = ''.join(gram)
            counts[gstr] = counts.get(gstr, 0) + 1
            if gstr not in positions:
                positions[gstr] = []
            positions[gstr].append(i)
        # 收集重复项
        for gstr, c in counts.items():
            if c > 1:
                repetitive_ngrams.append(gstr)
        stats[n] = {
            'counts': counts,
            'positions': positions,
        }
    return repetitive_ngrams, stats


def batch_find_repetitive_ngram(tokens_batch, n_min=2, n_max=4, max_workers: int | None = None):
    """
    并行批量版本：对多个样本的 token 序列统计重复 n-gram。

    参数：
    - tokens_batch: List[List[str]]，每个元素为一个样本的 token 列表。
    - n_min / n_max: 与单样本版本含义相同。
    - max_workers: 并行线程数，None 表示使用默认线程数。

    返回：
    - repetitive_ngrams_batch: List[List[str]]，每个样本的重复 n-gram 字符串列表。
    - stats_batch: List[dict]，每个样本的统计字典（结构同单样本的 stats）。
    """
    if not isinstance(tokens_batch, (list, tuple)):
        raise ValueError("tokens_batch 应为 List[List[str]] 或 Tuple[List[str]]")

    def _worker(tokens):
        return find_repetitive_ngram(tokens, n_min=n_min, n_max=n_max)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_worker, tokens_batch))

    repetitive_ngrams_batch = [r[0] for r in results]
    stats_batch = [r[1] for r in results]
    return repetitive_ngrams_batch, stats_batch
    

# rep-w, counting the current token occurs in previous prefix (overall prefix sequence) in the generations of the test set
def calculate_rep_w(text_list, w=16):
    # code borrowed from the zihaofu's paper: A Theoretical Analysis of the Repetition Problem in Text Generation
    # tokens are the BPE tokens from this paper: NEURAL TEXT DEGENERATION WITH UNLIKELIHOOD TRAINING
    rep_w = []
    for text in tqdm(text_list):
        tokens = text.split()
        rep_w_single = 0
        for idx in range(1, len(tokens)):
            t = tokens[idx]
            prefix = set(tokens[max(0, idx-w):idx])
            if t in prefix:
                rep_w_single += 1
        if len(tokens) <= 1:
            continue
        rep_w_single /= len(tokens) - 1
        rep_w.append(rep_w_single)
    rep_w = np.mean(rep_w) * 100
    return rep_w

# code borrowed from the zihaofu's paper: A Theoretical Analysis of the Repetition Problem in Text Generation
# https://github.com/fuzihaofzh/repetition-problem-nlg/blob/f0f80ea986d288fb5a76f48d4d16ddb60cace575/src/eval_metrics.py#L133

def calculate_rep_r(text_list):
    rep_r_list = []
    for text in tqdm(text_list):
        tokens = text.split()
        if len(tokens) < 2:
            rep_r_list.append(0)
        counter = {}
        for j in range(len(tokens) - 1):
            gm = ' '.join(tokens[j : j + 2])
            counter[gm] = counter[gm] + 1 if gm in counter else 1
        label = [0] * len(tokens)
        for i in range(1, len(tokens)):
            if counter['%s %s'%(tokens[i-1], tokens[i])] > 1:
                label[i-1] = label[i] = 1         
        try:
            ratio = sum(label) / len(label)
            rep_r_list.append(ratio)
        except:
            pass
    rep_r = np.mean(rep_r_list) * 100
    return rep_r


def compute_repetition_ratio(text_list):
    ngram_list = [2,3,4]
    results = {i: {'num_rep': [], 'num_total': []} for i in ngram_list}
    for text in tqdm(text_list):
        if text is None:
            print(ptr)
        rest_dict = compute_instance(text, ngram_list)
        for n, (num_rep, num_total) in rest_dict.items():
            results[n]['num_rep'].append(num_rep)
            results[n]['num_total'].append(num_total)
    final = {i: -1 for i in ngram_list}
    for n, item in results.items():
        a = sum(item['num_rep'])
        b = sum(item['num_total'])
        
        final[n] = round(100 * a/b, 4)
    return final

def compute_instance(text, ngram_list):
    res_dict = {}
    for n in ngram_list:
        num_rep, num_total = eval_text(text, n)
        res_dict[n] = (num_rep, num_total)
    return res_dict

def eval_text(text, ngram):
    token_list = text.strip().split()
    ngram_list = list(ngrams(token_list, ngram))
    ngram_set = set()
    counter = 0
    for item in ngram_list:
        if item not in ngram_set:
            ngram_set.add(item)
        else:
            counter += 1
    if len(ngram_list) > 0:
        return counter, len(ngram_list)
    else:
        return 0, 0



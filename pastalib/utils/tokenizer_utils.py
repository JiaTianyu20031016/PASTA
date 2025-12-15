"""Utils for interacting with huggingface tokenizers."""
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Sequence, Tuple

from .typing import StrSequence, Tokenizer, TokenizerOffsetMapping


def find_token_range(
    string: str,
    substring: str,
    tokenizer: Optional[Tokenizer] = None,
    occurrence: int = 0,
    offset_mapping: Optional[TokenizerOffsetMapping] = None,
    **kwargs: Any,
) -> Tuple[int, int] | Sequence[Tuple[int, int]]:
    """Find index range of tokenized string containing tokens for substring.

    The kwargs are forwarded to the tokenizer.

    A simple example:

        string = 'The batman is the night.'
        substring = 'batman'
        tokenizer = ...

        # Example tokenization: ['the', 'bat', '##man', 'is', 'the', 'night']
        assert find_token_range(string, substring, tokenizer) == (1, 3)

    Args:
        string: The string.
        substring: The substring to find token range for.
        tokenizer: The tokenizer. If not set, offset_mapping must be.
        occurrence: The occurence of the substring to look for.
            Zero indexed. Defaults to 0, the first occurrence.
        offset_mapping: Precomputed offset mapping. If not set, tokenizer will be run.

    Raises:
        ValueError: If substring is not actually in string or if banned
            kwargs are specified.

    Returns:
        - 当 occurrence >= 0：返回单个范围 `Tuple[int, int]`（起始含、结束不含）。
        - 当 occurrence == -1：返回所有出现的范围 `List[Tuple[int, int]]`。
    """
    if tokenizer is None and offset_mapping is None:
        raise ValueError("must set either tokenizer= or offset_mapping=")
    if "return_offsets_mapping" in kwargs:
        raise ValueError("cannot set return_offsets_mapping")
    if substring not in string:
        raise ValueError(f'"{substring}" not found in "{string}"')

    if len(substring) == 0:
        return (0, 0)
    
    # 获取 offset mapping（若未提供则通过 tokenizer 计算）
    if offset_mapping is None:
        assert tokenizer is not None
        tokens = tokenizer(string, return_offsets_mapping=True, **kwargs)
        offset_mapping = tokens.offset_mapping

    # occurrence == -1: 返回所有匹配的 token 范围
    if occurrence == -1:
        all_ranges: list[Tuple[int, int]] = []
        search_pos = 0
        while True:
            idx = string.find(substring, search_pos)
            if idx == -1:
                break
            char_start = idx
            char_end = idx + len(substring)
            token_start, token_end = None, None
            for index, (token_char_start, token_char_end) in enumerate(offset_mapping):
                if token_start is None:
                    if token_char_start <= char_start and token_char_end >= char_start:
                        token_start = index
                if token_end is None:
                    if token_char_start <= char_end and token_char_end >= char_end:
                        token_end = index
                        break
            assert token_start is not None
            assert token_end is not None
            assert token_start <= token_end
            all_ranges.append((token_start, token_end + 1))
            search_pos = char_start + 1
        # 合并重叠范围（基于 token 索引）
        if not all_ranges:
            return all_ranges
        all_ranges.sort(key=lambda x: (x[0], x[1]))
        merged: list[Tuple[int, int]] = []
        cur_start, cur_end = all_ranges[0]
        for s, e in all_ranges[1:]:
            if s <= cur_end:  # 重叠或相邻（相邻也视为无重叠，保留分隔）；这里仅合并重叠：s <= cur_end-1
                # 合并到当前区间
                if e > cur_end:
                    cur_end = e
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = s, e
        merged.append((cur_start, cur_end))
        return merged

    # occurrence >= 0: 返回指定出现的范围
    char_start = string.index(substring)
    for _ in range(occurrence):
        try:
            char_start = string.index(substring, char_start + 1)
        except ValueError as error:
            raise ValueError(
                f"could not find {occurrence + 1} occurrences "
                f'of "{substring} in "{string}"'
            ) from error
    char_end = char_start + len(substring)

    token_start, token_end = None, None
    for index, (token_char_start, token_char_end) in enumerate(offset_mapping):
        if token_start is None:
            if token_char_start <= char_start and token_char_end >= char_start:
                token_start = index
        if token_end is None:
            if token_char_start <= char_end and token_char_end >= char_end:
                token_end = index
                break

    assert token_start is not None
    assert token_end is not None
    assert token_start <= token_end
    return (token_start, token_end + 1)


def batch_convert_ids_to_tokens(
    batch: Sequence[Sequence[int]], tokenizer: Tokenizer, **kwargs: Any
) -> Sequence[StrSequence]:
    """Call `convert_ids_to_tokens` on every sequence in the batch."""
    return [tokenizer.convert_ids_to_tokens(ids, **kwargs) for ids in batch]


@contextmanager
def set_padding_side(
    tokenizer: Tokenizer, padding_side: str = "right"
) -> Iterator[None]:
    """Temporarily set padding side for tokenizer.

    Useful for when you want to batch generate with causal LMs like GPT, as these
    require the padding to be on the left side in such settings but are much easier
    to mess around with when the padding, by default, is on the right.

    Example usage:
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        with tokenizer_utils.set_padding_side(tokenizer, "left"):
            inputs = mt.tokenizer(...)
        # ... later
        model.generate(**inputs)

    """
    _padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    yield
    tokenizer.padding_side = _padding_side

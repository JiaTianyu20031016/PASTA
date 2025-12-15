"""PASTA Implementation"""
import torch
import abc, json
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, cast, overload, Tuple
import time

import transformers 
from pastalib.utils import tokenizer_utils, repetition_utils, model_utils
from pastalib.utils.typing import (
    Model,
    Dataset,
    Device,
    ModelInput,
    ModelOutput,
    StrSequence,
    Tokenizer,
    TokenizerOffsetMapping,
)


class PASTA(abc.ABC):
    """
    Create PASTA to steer attentions of transformer models. 

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be steered. 
        tokenizer ([`transformers.PreTrainedTokenizer`]): The model's tokenizer. 
        head_config (`dict`): The config to control which attention heads to be steered. 
        alpha (`float`): The scaling coefficient of attention steering. 
        scale_position (`str`): To upweight the scores of highlighted tokens (`include`), 
            or to downweight those of unselected tokens (`exclude`). 

    Returns:
        PASTA: PASTA steerer that can register the pre-forward hooks on target models. 

    """

    ATTN_MODULE_NAME = {
        "gptj": "transformer.h.{}.attn",
        "gpt2": "transformer.h.{}.attn",
        "llama": "model.layers.{}.self_attn",
        "mistral": "model.layers.{}.self_attn",
        "gemma": "model.layers.{}.self_attn",
        "phi3mini": "model.layers.{}.self_attn"
    }
    ATTENTION_MASK_ARGIDX = {
        "gptj": 2, 
        "gpt2": 3,
        "llama": 1, 
        "mistral": 1, 
        "gemma": 1,
    }
    MODEL_MASK_ARGIDX = {
        "gptj": 2, 
        "gpt2": 3,
        "llama": 1, 
        "mistral": 1, 
        "gemma": 1,
    }
    MODEL_INPUT_ID_ARGIDX = {
        "gptj": 0, 
        "gpt2": 0,
        "llama": 0, 
        "mistral": 0, 
        "gemma": 0,
    }
    def __init__(
        self, 
        model: Model, 
        tokenizer: Tokenizer, 
        head_config: dict|list|None = None, 
        alpha: float = 0.01, 
        scale_position: str = "exclude", 
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.setup_model(model)

        self.alpha = alpha
        self.scale_position = scale_position
        self.setup_head_config(head_config)

        assert self.scale_position in ['include', 'exclude', 'generation']
        assert self.alpha > 0
        self.scale_constant = None
        # 动态注册的子模块 hooks（在一次前向结束后清理）
        self._dynamic_registered_hooks: list[Any] = []
        # 缓存用于生成模式下逐 token 追加的 input_ids / attention_mask
        self._cached_input_ids: torch.Tensor | None = None
        self._cached_attention_mask: torch.Tensor | None = None

    def setup_model(self, model):
        """Obtain the model type and complete the configuration."""
        if isinstance(model, transformers.LlamaForCausalLM):
            self.model_name = "llama"
            self.num_attn_head = model.config.num_attention_heads
        elif isinstance(model, transformers.GPTJForCausalLM):
            self.model_name = "gptj"
            self.num_attn_head = model.config.n_head
        elif isinstance(model, transformers.GPT2LMHeadModel):
            self.model_name = "gpt2"
            self.num_attn_head = model.config.n_head
        elif isinstance(model, transformers.MistralForCausalLM):
            self.model_name = "mistral"
            self.num_attn_head = model.config.num_attention_heads
        elif isinstance(model, transformers.GemmaForCausalLM):
            self.model_name = "gemma"
            self.num_attn_head = model.config.num_attention_heads
        elif model.__class__.__name__ == "Phi3ForCausalLM":
            self.model_name = "phi3mini"
            self.num_attn_head = model.config.num_attention_heads
        else:
            raise ValueError("Unimplemented Model Type.")
        
    def setup_head_config(self, head_config):
        """
        Config the attention heads to be steered.

        If `head_config` is `list` of layer index, PASTA will steer the entire layers. 
        """
        if isinstance(head_config, dict):
            self.head_config = {int(k):v for k,v in head_config.items()} 
            self.all_layers_idx = [int(key) for key in head_config]
        elif isinstance(head_config, list):
            self.all_layers_idx = [int(v) for v in head_config]
            self.head_config = {
                idx:list(range(self.num_attn_head)) for idx in self.all_layers_idx
            }
        else:
            raise ValueError(f"Incorrect head config: {head_config}")
    
    def _maybe_batch(self, text: str | StrSequence) -> StrSequence:
        """Batch the text if it is not already batched."""
        if isinstance(text, str):
            return [text]
        return text

    def token_ranges_from_batch(
        self,
        strings: str | StrSequence,
        substrings: str | StrSequence,
        offsets_mapping: Sequence[TokenizerOffsetMapping],
        occurrence: int = 0,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Return token ranges for (str, substr) pairs.

        - 当 `occurrence >= 0`：返回形状 (batch_size, 2) 的单个 `torch.Tensor`。
        - 当 `occurrence == -1`：`find_token_range` 返回每对 (string, substring) 的多个范围，
          本函数将对每个样本的范围列表进行长度对齐（用 (0, 0) 右侧补齐到该批次的最大长度），
          并返回 `List[torch.Tensor]`，其中每个 `Tensor` 形状为 (batch_size, 2)。
        """
        strings = self._maybe_batch(strings)
        substrings = self._maybe_batch(substrings)
        if len(strings) != len(substrings):
            raise ValueError(
                f"got {len(strings)} strings but only {len(substrings)} substrings"
            )
        # 简单模式：单一 occurrence
        if occurrence >= 0:
            return torch.tensor(
                [
                    tokenizer_utils.find_token_range(
                        string, substring, offset_mapping=offset_mapping, occurrence=occurrence
                    )
                    for string, substring, offset_mapping in zip(strings, substrings, offsets_mapping)
                ]
            )

        # 复杂模式：occurrence == -1，需要对每个样本的多段范围进行对齐并分发
        # 收集每个样本的范围列表
        per_sample_ranges: list[list[tuple[int, int]]] = []
        for string, substring, offset_mapping in zip(strings, substrings, offsets_mapping):
            ranges = tokenizer_utils.find_token_range(
                string, substring, offset_mapping=offset_mapping, occurrence=occurrence
            )
            if isinstance(ranges, tuple):
                ranges = [ranges]
            # ranges 为 List[Tuple[int,int]]
            per_sample_ranges.append(ranges)

        # 计算该批次的最大段数
        max_len = max((len(r) for r in per_sample_ranges), default=0)
        if max_len == 0:
            # 全为空，返回一个空列表
            return []

        # 逐段构建 (batch_size, 2) 的张量列表；不足的样本用 (0,0) 填充
        batch_tensors: list[torch.Tensor] = []
        bsz = len(per_sample_ranges)
        for seg_idx in range(max_len):
            seg_ranges: list[tuple[int, int]] = []
            for rlist in per_sample_ranges:
                if seg_idx < len(rlist):
                    seg_ranges.append(rlist[seg_idx])
                else:
                    seg_ranges.append((0, 0))
            batch_tensors.append(torch.tensor(seg_ranges))

        return batch_tensors


    def edit_attention_mask(
        self, 
        module: torch.nn.Module, 
        input_args: tuple,
        input_kwargs: dict, 
        head_idx: list[int],
        token_range: torch.Tensor, 
        input_len: int, 
    ):
        """
        The hook function registerred pre-forward for attention models. 

        Args: 
            module ([`torch.nn.Module`]): The registerred attention modules. 
            input_args (`tuple`): The positional arguments of forward function. 
            input_kwargs (`dict`): The keyword arguments of forward function. 
            head_idx (`list[int]`): The index of heads to be steered. 
            token_range (`torch.Tensor`): A B*2 tensor, 
                suggesting the index range of hightlight tokens.  
            input_len (`int`): The length L of inputs.

        Returns: 
            tuple, dict: return the modified `attention_mask`,
                while not changing other input arguments. 
        """
        if "attention_mask" in input_kwargs:
            attention_mask = input_kwargs['attention_mask'].clone()
        elif input_args is not None:
            arg_idx = self.ATTENTION_MASK_ARGIDX[self.model_name]
            attention_mask = input_args[arg_idx].clone()
        else:
            raise ValueError(f"Not found attention masks in {str(module)}")
        
        bsz, head_dim, tgt_len, src_len = attention_mask.size()
        dtype, device = attention_mask.dtype, attention_mask.device
        if head_dim != self.num_attn_head:
            attention_mask = attention_mask.expand(
                bsz, self.num_attn_head, tgt_len, src_len
            ).clone()
        if not self.scale_constant:
            self.scale_constant = torch.Tensor([self.alpha]).to(dtype).to(device).log()
        
        for bi, (ti,tj) in enumerate(token_range.tolist()):
            if self.scale_position == "include":
                attention_mask[bi, head_idx, :, ti:tj] += self.scale_constant
            else:
                attention_mask[bi, head_idx, :, :ti] += self.scale_constant
                attention_mask[bi, head_idx, :, tj:input_len] += self.scale_constant
        
        if self.model_name in ["llama", "mistral", "gemma", "phi3mini"]:
            attention_mask.old_size = attention_mask.size 
            attention_mask.size = lambda:(bsz, 1, tgt_len, src_len)
        
        if "attention_mask" in input_kwargs:
            input_kwargs['attention_mask'] = attention_mask 
            return input_args, input_kwargs
        else:
            return (input_args[:arg_idx], attention_mask, *input_args[arg_idx+1:]), input_kwargs

    def edit_multisection_attention(
        self, 
        module: torch.nn.Module, 
        input_args: tuple,
        input_kwargs: dict, 
        head_idx: list[int],
        token_ranges: list[torch.Tensor], 
        input_len: int, 
    ):
        """
        The hook function registerred pre-forward for attention models. 

        Args: 
            module ([`torch.nn.Module`]): The registerred attention modules. 
            input_args (`tuple`): The positional arguments of forward function. 
            input_kwargs (`dict`): The keyword arguments of forward function. 
            head_idx (`list[int]`): The index of heads to be steered. 
            token_ranges (`torch.Tensor`): A list of B*2 tensors, 
                suggesting the index range of hightlight tokens of multiple sections.  
            input_len (`int`): The length L of inputs.

        Returns: 
            tuple, dict: return the modified `attention_mask`,
                while not changing other input arguments. 
        """
        if "attention_mask" in input_kwargs:
            attention_mask = input_kwargs['attention_mask'].clone().detach()
        elif input_args is not None:
            arg_idx = self.ATTENTION_MASK_ARGIDX[self.model_name]
            attention_mask = input_args[arg_idx].clone().detach()
        else:
            raise ValueError(f"Not found attention masks in {str(module)}")
        
        bsz, head_dim, tgt_len, src_len = attention_mask.size()
        dtype, device = attention_mask.dtype, attention_mask.device
        if head_dim != self.num_attn_head:
            attention_mask = attention_mask.expand(
                bsz, self.num_attn_head, tgt_len, src_len
            ).clone().to(dtype).to(device)
        if not self.scale_constant:
            self.scale_constant = torch.Tensor([self.alpha]).to(dtype).to(device).log()
        
        for token_range in token_ranges:
            for bi, (ti,tj) in enumerate(token_range.tolist()):
                # ignore place-holders
                if ti == tj == 0:
                    continue
                if self.scale_position == "include":
                    attention_mask[bi, head_idx, :, ti:tj] += self.scale_constant
                elif self.scale_position == "exclude":
                    attention_mask[bi, head_idx, :, :ti] += self.scale_constant
                    attention_mask[bi, head_idx, :, tj:input_len] += self.scale_constant
                elif self.scale_position == "generation":
                    attention_mask[bi, head_idx, :, :input_len] += self.scale_constant 
                else:
                    raise ValueError(f"Unexcepted {self.scale_position}.")
        if self.scale_position == "include":
            attention_mask[:, head_idx, :, :input_len] -= self.scale_constant
        
        if self.model_name in ["llama", "mistral", "phi3mini"]:
            attention_mask.old_size = attention_mask.size 
            attention_mask.size = lambda:(bsz, 1, tgt_len, src_len)
        
        if "attention_mask" in input_kwargs:
            input_kwargs['attention_mask'] = attention_mask 
            return input_args, input_kwargs
        else:
            return (input_args[:arg_idx], attention_mask, input_args[arg_idx+1:]), input_kwargs


    def build_operation_masks(
        self,
        token_ranges: list[torch.Tensor],
        head_config: dict[int, list[int]] | None = None,
        input_len: int | None = None,
        bsz: int | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> dict[int, torch.Tensor]:
        """Vectorized construction of per-layer masks with scale_position applied.

        返回的掩码已经融合了 scale_position 语义，并乘上 ``log(alpha)``，可直接加到
        attention_mask 上。形状：``(bsz, num_heads, 1, input_len)``。
        """
        if head_config is None:
            head_config = self.head_config
        if bsz is None:
            bsz = token_ranges[0].size(0)
        else:
            assert bsz == token_ranges[0].size(0)

        if not token_ranges:
            raise ValueError("No token ranges provided for building operation masks.")

        if input_len is None:
            raise ValueError("input_len must be provided to build_operation_masks.")

        if dtype is None:
            dtype = token_ranges[0].dtype
        if device is None:
            device = token_ranges[0].device

        if self.scale_constant is None:
            self.scale_constant = torch.tensor([self.alpha], device=device, dtype=dtype).log()

        stacked = torch.stack(token_ranges, dim=0)  # (num_sections, bsz, 2)
        starts = stacked[..., 0]
        ends = stacked[..., 1]
        valid = ends > starts

        positions = torch.arange(input_len, device=device).view(1, 1, input_len)
        section_mask = (positions >= starts.unsqueeze(-1)) & (positions < ends.unsqueeze(-1))
        section_mask = section_mask & valid.unsqueeze(-1)

        if self.scale_position == "include":
            base_counts = section_mask.sum(dim=0).to(dtype) - 1
        elif self.scale_position == "exclude":
            outside_mask = valid.unsqueeze(-1) & (~section_mask)
            base_counts = outside_mask.sum(dim=0).to(dtype)
        elif self.scale_position == "generation":
            valid_counts = valid.sum(dim=0).to(dtype)
            base_counts = valid_counts.unsqueeze(-1).expand(-1, input_len)
        else:
            raise ValueError(f"Unexpected scale_position {self.scale_position}.")

        base_mask = base_counts * self.scale_constant.view(1)
        
        op_masks: dict[int, torch.Tensor] = {}
        head_template = torch.zeros(self.num_attn_head, device=device, dtype=dtype)
        for layer_idx, heads in head_config.items():
            head_weights = head_template.clone()
            head_weights[heads] = 1.0
            # (bsz, num_heads, 1, input_len) – target dim is expanded in the caller.
            op_masks[layer_idx] = base_mask.unsqueeze(1).unsqueeze(2) * head_weights.view(1, -1, 1, 1)

        return op_masks

    def edit_multisection_attention_fast(
        self, 
        module: torch.nn.Module, 
        input_args: tuple,
        input_kwargs: dict, 
        operation_mask: torch.Tensor,
        input_len: int, 
        token_ranges: list[torch.Tensor]=None, 
        head_idx: list[int]=None,
    ):
        """
        The hook function registerred pre-forward for attention models. 

        Args: 
            module ([`torch.nn.Module`]): The registerred attention modules. 
            input_args (`tuple`): The positional arguments of forward function. 
            input_kwargs (`dict`): The keyword arguments of forward function. 
            head_idx (`list[int]`): The index of heads to be steered. 
            token_ranges (`torch.Tensor`): A list of B*2 tensors, 
                suggesting the index range of hightlight tokens of multiple sections.  
            input_len (`int`): The length L of inputs.

        Returns: 
            tuple, dict: return the modified `attention_mask`,
                while not changing other input arguments. 
        """
        if "attention_mask" in input_kwargs:
            attention_mask = input_kwargs['attention_mask'].clone().detach()
        elif input_args is not None:
            arg_idx = self.ATTENTION_MASK_ARGIDX[self.model_name]
            attention_mask = input_args[arg_idx].clone().detach()
        else:
            raise ValueError(f"Not found attention masks in {str(module)}")
        
        bsz, head_dim, tgt_len, src_len = attention_mask.size()
        dtype, device = attention_mask.dtype, attention_mask.device
        if head_dim != self.num_attn_head:
            attention_mask = attention_mask.expand(
                bsz, self.num_attn_head, tgt_len, src_len
            ).clone().to(dtype).to(device)
        if self.scale_constant is None:
            self.scale_constant = torch.Tensor([self.alpha]).to(dtype).to(device).log()
        
        mask = operation_mask.to(device=device, dtype=dtype)
        if mask.dim() == 3:
            mask = mask.unsqueeze(2)

        # head_filter = torch.zeros(self.num_attn_head, device=device, dtype=dtype)
        # head_filter[head_idx] = 1.0
        # mask = mask * head_filter.view(1, -1, 1, 1)

        if mask.size(2) == 1 and tgt_len > 1:
            mask = mask.expand(-1, -1, tgt_len, -1)

        if mask.size(-1) != src_len:
            raise ValueError(
                f"operation_mask last dim {mask.size(-1)} mismatches attention src_len {src_len}"
            )

        temp_attention_mask = attention_mask.clone()
        temp_attention_mask = temp_attention_mask + mask
        for token_range in token_ranges:
            for bi, (ti,tj) in enumerate(token_range.tolist()):
                # ignore place-holders
                if ti == tj == 0:
                    continue
                if self.scale_position == "include":
                    attention_mask[bi, head_idx, :, ti:tj] += self.scale_constant
                elif self.scale_position == "exclude":
                    attention_mask[bi, head_idx, :, :ti] += self.scale_constant
                    attention_mask[bi, head_idx, :, tj:input_len] += self.scale_constant
                elif self.scale_position == "generation":
                    attention_mask[bi, head_idx, :, :input_len] += self.scale_constant 
                else:
                    raise ValueError(f"Unexcepted {self.scale_position}.")
        if self.scale_position == "include":
            attention_mask[:, head_idx, :, :input_len] -= self.scale_constant
        assert torch.allclose(attention_mask, temp_attention_mask), \
            "fast attention mask does not match the original one"

        if self.model_name in ["llama", "mistral", "phi3mini"]:
            attention_mask.old_size = attention_mask.size 
            attention_mask.size = lambda:(bsz, 1, tgt_len, src_len)
        
        if "attention_mask" in input_kwargs:
            input_kwargs['attention_mask'] = attention_mask 
            return input_args, input_kwargs
        else:
            return (input_args[:arg_idx], attention_mask, input_args[arg_idx+1:]), input_kwargs


    @contextmanager
    def apply_steering(
        self, 
        model: Model, 
        strings: list, 
        substrings: list[list[str]], 
        model_input: ModelInput, 
        offsets_mapping: Sequence[TokenizerOffsetMapping], 
        occurrence: int = 0,
    ):
        """
        The function of context manager to register the pre-forward hook on `model`. 

        Args:
            model ([`transformers.PreTrainedModel`]): The transformer model to be steered. 
            strings (`list[str]`): The input strings. 
            substrings (`list[list[str]]` or list[str]): The highlighted input spans for each string. 
            model_input (`transformers.BatchEncoding`): The batched model inputs. 
            offsets_mapping (`TokenizerOffsetMapping`): The offset mapping outputed by
                the tokenizer when encoding the `strings`. 
        """

        
        if isinstance(substrings[0], str):
            substrings = [substrings]
        assert len(strings) == len(substrings) == len(offsets_mapping), \
            f"got {len(strings)} strings, {len(substrings)} substrings, "\
            f"and {len(offsets_mapping)} offsets_mapping"

        # substrings: List[List[str]]，每个样本的若干子串；需要补齐并转置
        # 计算最大段数
        max_len = max((len(ss) for ss in substrings), default=0)
        # 用空字符串补齐到等长
        padded_substrings: list[list[str]] = []
        for ss in substrings:
            if len(ss) < max_len:
                ss = ss + [""] * (max_len - len(ss))
            padded_substrings.append(ss)
        # 转置：得到每一段的跨样本列表
        transposed_sections: list[list[str]] = []
        if max_len > 0:
            for idx in range(max_len):
                transposed_sections.append([sample_sections[idx] for sample_sections in padded_substrings])

        token_ranges = []
        for sections in transposed_sections:
            token_range = self.token_ranges_from_batch(
                strings, sections, offsets_mapping, occurrence=occurrence,
            )
            if isinstance(token_range, torch.Tensor):
                token_ranges.append(token_range)
            else:
                token_ranges.extend(token_range)

        registered_hooks = []
        for layer_idx in self.all_layers_idx:
            name = self.ATTN_MODULE_NAME[self.model_name].format(layer_idx)
            module = model.get_submodule(name)
            # Prepare the hook function with partial arguments being fixed. 
            # Pass the head_idx, token_range, input_len for each attention module in advance. 
            hook_func = partial(
                self.edit_multisection_attention, 
                head_idx = self.head_config[layer_idx],
                token_ranges = token_ranges, 
                input_len = model_input['input_ids'].size(-1)
            )
            registered_hook = module.register_forward_pre_hook(hook_func, with_kwargs=True)
            registered_hooks.append(registered_hook)
        try:
            yield model
        except Exception as error:
            raise error
        finally:
            for registered_hook in registered_hooks:
                registered_hook.remove()
    

    def inputs_from_batch(
        self, 
        text: str | StrSequence,
        tokenizer: Tokenizer|None = None,
        device: Optional[Device] = None,
    ) -> tuple[ModelInput, Sequence[TokenizerOffsetMapping]]:
        """Precompute model inputs."""
        if tokenizer is None:
            tokenizer = self.tokenizer
        with tokenizer_utils.set_padding_side(tokenizer, padding_side="left"):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=False,
                padding="longest",
                return_offsets_mapping=True,
            )
            offset_mapping = inputs.pop("offset_mapping")
        if device is not None:
            inputs = inputs.to(device)
        return inputs, offset_mapping


    @classmethod
    def load_head_config(cls, file:str|Path):
        """Load the `head_config` from JSON file."""
        with open(file, "r") as f:
            head_config = json.load(f)
        return head_config 


    def token_ranges_from_model_input(
        self,
        model_input: ModelInput,
        n_min: int = 2,
        n_max: int = 4,
    ) -> list[torch.Tensor]:
        """
        基于输入 batch 的重复 n-gram（忽略 padding）生成 token ranges。

        返回：List[Tensor]，每个张量形状为 (batch_size, 2)。不足的样本位置用 (0, 0) 占位。
        """
        input_ids = model_input["input_ids"]
        attention_mask = model_input.get("attention_mask", None)
        pad_id = self.tokenizer.pad_token_id

        tokens_batch = []
        # 记录从去 padding 的序列到原始序列的左偏移，以便将范围映射回原始索引
        left_offsets: list[int] = []
        for i in range(input_ids.size(0)):
            ids = input_ids[i]
            left_offset = 0
            if attention_mask is not None:
                mask = attention_mask[i].bool()
                # 仅考虑有效 token（通常为中间连续区间）；计算左端第一个有效位置作为偏移
                if mask.any():
                    left_offset = int(torch.argmax(mask.int()))
                ids = ids[mask]
            elif pad_id is not None:
                # 不假定 pad 在左或右，两侧同时剔除 pad_id，并记录左侧剔除数量作为偏移
                arr = ids.tolist()
                left = 0
                right = len(arr)
                while left < right and arr[left] == pad_id:
                    left += 1
                while right > left and arr[right - 1] == pad_id:
                    right -= 1
                left_offset = left
                ids = ids[left:right]
            tokens = self.tokenizer.convert_ids_to_tokens(ids.tolist())
            tokens_batch.append(tokens)
            left_offsets.append(left_offset)

        # 统计重复 n-gram
        _, stats_batch = repetition_utils.batch_find_repetitive_ngram(
            tokens_batch, n_min=n_min, n_max=n_max
        )

        per_sample_ranges: list[list[tuple[int, int]]] = []
        for sample_idx, stats in enumerate(stats_batch):
            ranges: list[tuple[int, int]] = []
            for n in range(n_min, n_max + 1):
                positions = stats.get(n, {}).get("positions", {})
                counts = stats.get(n, {}).get("counts", {})
                for ngram_str, pos_list in positions.items():
                    if counts.get(ngram_str, 0) <= 1:
                        continue
                    for start_idx in pos_list:
                        # 将去 padding 后的索引映射回原始 input_ids 的索引
                        orig_start = start_idx + left_offsets[sample_idx]
                        ranges.append((orig_start, orig_start + n))
            # 排序并合并重叠区间
            ranges.sort(key=lambda x: (x[0], x[1]))
            merged: list[tuple[int, int]] = []
            for seg in ranges:
                if not merged or seg[0] > merged[-1][1]:
                    merged.append(seg)
                else:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], seg[1]))
            per_sample_ranges.append(merged)

        max_len = max((len(r) for r in per_sample_ranges), default=0)
        if max_len == 0:
            return []

        batch_tensors: list[torch.Tensor] = []
        for idx in range(max_len):
            seg: list[tuple[int, int]] = []
            for rlist in per_sample_ranges:
                if idx < len(rlist):
                    seg.append(rlist[idx])
                else:
                    seg.append((0, 0))
            batch_tensors.append(torch.tensor(seg))

        return batch_tensors


    def dynamic_edit_multisection_attention(
        self, 
        module: torch.nn.Module, 
        input_args: tuple,
        input_kwargs: dict, 
        n_min: int,
        n_max: int,
        mode: Literal['default', 'fast']='default',
    ):
        """
        注册在整个模型上的 pre-forward hook：
        - 从 model.forward 的参数中读取 model_input（如 input_ids, attention_mask）。
        - 调用 token_ranges_from_model_input 计算重复 n-gram 的 token ranges（基于原始索引）。
        - 为用户指定的注意力子模块注册 edit_multisection_attention 作为 pre-forward hooks。
        - 在本次前向结束后，自动清理这些临时 hooks。
        """

        # 1) 从 forward 参数构造 model_input
        # 支持 input_kwargs 优先，其次解析 input_args 的常见位置顺序 (input_ids, attention_mask, ...)
        model_input: dict[str, torch.Tensor] = {}
        if isinstance(input_kwargs, dict) and 'input_ids' in input_kwargs:
            model_input['input_ids'] = input_kwargs['input_ids']
        elif isinstance(input_args, tuple) and len(input_args) > 0 and isinstance(input_args[0], torch.Tensor):
            arg_idx = self.MODEL_INPUT_ID_ARGIDX[self.model_name]
            model_input['input_ids'] = input_args[arg_idx]
        else:
            raise ValueError(f"Not found input_ids in model forward args of {str(module)}")
        if 'attention_mask' in input_kwargs:
            model_input['attention_mask'] = input_kwargs['attention_mask']
        elif isinstance(input_args, tuple) and len(input_args) > 1 and isinstance(input_args[1], torch.Tensor):
            arg_idx = self.MODEL_MASK_ARGIDX[self.model_name]
            model_input['attention_mask'] = input_args[arg_idx]
        else:
            pass  # attention_mask 可选

        # 兼容 generate 增量调用：当只传入单步 input_ids 时，将其与缓存拼接，构造完整序列
        cur_ids = model_input['input_ids']
        cur_mask = model_input.get('attention_mask', None)
        # 如果没有缓存，初始化缓存
        if self._cached_input_ids is None:
            self._cached_input_ids = cur_ids.detach()
        # 如果收到的仅是新增 token（通常 shape [B,1]），则与缓存拼接形成完整序列
        elif cur_ids.size(-1) == 1:
            full_ids = torch.cat([self._cached_input_ids.to(cur_ids.device), cur_ids], dim=-1)
            model_input['input_ids'] = full_ids
            self._cached_input_ids = full_ids.detach()
        else:
            # 收到完整序列，刷新缓存
            self._cached_input_ids = cur_ids.detach()

        if cur_mask is not None:
            if self._cached_attention_mask is None:
                self._cached_attention_mask = cur_mask.detach()
            elif cur_mask.size(-1) == 1:
                full_mask = torch.cat([self._cached_attention_mask.to(cur_mask.device), cur_mask], dim=-1)
                model_input['attention_mask'] = full_mask
                self._cached_attention_mask = full_mask.detach()
            else:
                self._cached_attention_mask = cur_mask.detach()

        input_len = model_input['input_ids'].size(-1)
        # 2) 计算 ranges（合并重叠、映射原始索引）
        token_ranges = self.token_ranges_from_model_input(model_input, n_min=n_min, n_max=n_max)
        if mode == 'fast':
            # 预先构建各层的 operation_mask
            operation_masks = self.build_operation_masks(
                token_ranges,
                head_config=self.head_config,
                input_len=input_len,
                bsz=model_input['input_ids'].size(0),
                dtype=model_input['input_ids'].dtype,
                device=model_input['input_ids'].device,
            )

        # 3) 为每个目标注意力子模块注册 pre-forward hooks
        self._dynamic_registered_hooks.clear()
        for layer_idx in self.all_layers_idx:
            name = self.ATTN_MODULE_NAME[self.model_name].format(layer_idx)
            attn_module = module.get_submodule(name)
            if mode == 'fast':
                hook_func = partial(
                    self.edit_multisection_attention_fast,
                    head_idx=self.head_config[layer_idx],
                    #token_ranges=token_ranges,
                    operation_mask=operation_masks[layer_idx],
                    input_len=input_len,
                )
            else:
                hook_func = partial(
                    self.edit_multisection_attention,
                    head_idx=self.head_config[layer_idx],
                    token_ranges=token_ranges,
                    input_len=input_len,
                )
            registered_hook = attn_module.register_forward_pre_hook(hook_func, with_kwargs=True)
            self._dynamic_registered_hooks.append(registered_hook)

        # 4) 在模型上注册一次性的 forward hook，用于清理以上临时 hooks
        def _cleanup_hook(_mod: torch.nn.Module, _inp, _out):
            for h in self._dynamic_registered_hooks:
                try:
                    h.remove()
                except Exception:
                    pass
            self._dynamic_registered_hooks.clear()
            # 自身也移除
            try:
                cleanup_handle.remove()
            except Exception:
                pass

        cleanup_handle = module.register_forward_hook(_cleanup_hook)

        # 返回原始参数，不做修改
        return input_args, input_kwargs


    @contextmanager
    def dynamic_apply_steering(
        self,
        model: Model,
        n_min: int = 2,
        n_max: int = 4,
        mode: Literal['default', 'fast']='default',
    ):
        """
        模型级动态注意力引导的上下文管理器。

        进入上下文：在整个 `model` 上注册一次性的 pre-forward hook
        `dynamic_edit_multisection_attention`，它会在前向时自动：
        - 从 `model.forward` 参数采集 `model_input`；
        - 计算重复 n-gram 的 token ranges（原始索引，区间合并）；
        - 为指定层的注意力模块注册 `edit_multisection_attention` 子 hooks；
        - 在本次前向结束后清理这些子 hooks。

        退出上下文：移除模型级的动态 pre-hook。
        """
        # 注册模型级 pre-forward hook
        model_level_hook = model.register_forward_pre_hook(
            partial(self.dynamic_edit_multisection_attention, n_min=n_min, n_max=n_max, mode=mode),
            with_kwargs=True,
        )
        try:
            yield model
        except Exception as error:
            # 异常也尝试清理句柄
            try:
                model_level_hook.remove()
            except Exception:
                pass
            raise error
        finally:
            # 正常清理模型级 hook
            try:
                model_level_hook.remove()
            except Exception:
                pass


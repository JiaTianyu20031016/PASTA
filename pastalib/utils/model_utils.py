import re
from typing import Dict, List, Optional


def list_attention_heads(model) -> Dict[int, List[int]]:
	"""
	枚举给定 HuggingFace 模型中的所有注意力头，返回形如 {layer_id: [head_ids]} 的字典。

	适配策略：
	- 优先遍历 `model.named_modules()`，查找类名包含 "Attention" 的模块，读取其 `num_heads`/`n_heads`。
	- 尝试从模块名解析层索引（如 `h.0.attn`、`encoder.layer.3.attention`）。
	- 若遍历未命中，回退到 `model.config` 的 `num_hidden_layers`/`n_layer` 与 `num_attention_heads`/`n_head`。

	注意：不同模型的层级命名可能差异较大；本函数尽量泛化，无法完全覆盖所有自定义结构。
	"""

	def _get_config_layers_and_heads(m) -> tuple[Optional[int], Optional[int]]:
		cfg = getattr(m, "config", None)
		if cfg is None:
			return None, None
		num_layers = getattr(cfg, "num_hidden_layers", None)
		if num_layers is None:
			num_layers = getattr(cfg, "n_layer", None)
		num_heads = getattr(cfg, "num_attention_heads", None)
		if num_heads is None:
			num_heads = getattr(cfg, "n_head", None)
		return num_layers, num_heads

	def _parse_layer_id_from_name(name: str) -> Optional[int]:
		# 常见模式：h.<int>.attn / encoder.layer.<int>.attention / decoder.layers.<int>.self_attn
		tokens = name.split(".")
		for i, tok in enumerate(tokens):
			if tok in {"h", "layer", "layers"} and i + 1 < len(tokens):
				nxt = tokens[i + 1]
				if nxt.isdigit():
					return int(nxt)
		# 兜底：寻找最后一个数字片段
		m = re.findall(r"\.([0-9]+)(?:\.|$)", "." + name)
		if m:
			try:
				return int(m[-1])
			except Exception:
				pass
		return None

	result: Dict[int, List[int]] = {}
	seen_any = False
	auto_layer_counter = 0

	for name, module in model.named_modules():
		cls_name = module.__class__.__name__
		if "Attention" not in cls_name:
			continue
		# 读取 head 数
		num_heads = getattr(module, "num_heads", None)
		if num_heads is None:
			num_heads = getattr(module, "n_heads", None)
		if num_heads is None:
			# 某些实现可能没有显式字段，跳过该模块
			continue

		layer_id = _parse_layer_id_from_name(name)
		if layer_id is None:
			layer_id = auto_layer_counter
			auto_layer_counter += 1

		heads = list(range(int(num_heads)))
		existing = result.get(layer_id, [])
		# 合并并去重
		merged = sorted(set(existing) | set(heads))
		result[layer_id] = merged
		seen_any = True

	if seen_any:
		return dict(sorted(result.items(), key=lambda kv: kv[0]))

	# 遍历未找到时，回退到 config 信息
	cfg_layers, cfg_heads = _get_config_layers_and_heads(model)
	if cfg_layers is not None and cfg_heads is not None:
		return {i: list(range(int(cfg_heads))) for i in range(int(cfg_layers))}

	# 如果连 config 都不可用，返回空字典
	return {}


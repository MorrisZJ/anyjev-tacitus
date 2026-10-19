"""vLLM plugin: serve AnyJev Tacitus (Gemma 4 with bidirectional token ranges on its full-attention layers).

    pip install ./serving                    # registers the plugin; vLLM loads it in every process it starts
    ANYJEV_BIDI_GLOBAL=1 VLLM_USE_V2_MODEL_RUNNER=0 vllm serve <model> --attention-backend TRITON_ATTN \
        --limit-mm-per-prompt '{"image": 1, "video": 0}' --max-num-batched-tokens <longest prompt> ...

A client sends each prompt's bidirectional token ranges with the request, `vllm_xargs: {"bidi_ranges": [s0, e0, s1,
e1, ...]}` (inclusive positions; the AnyJev engine's `bidi=block` does this). vLLM 0.29 already runs per-request
bidirectional ranges for Gemma 4's image tokens (`mm_prefix_lm`), on the sliding layers; the plugin, active only with
ANYJEV_BIDI_GLOBAL=1, makes the same machinery serve text ranges on the full-attention layers:

  1. the range mode is on (ModelConfig.is_mm_prefix_lm), which also makes every attention layer pick a backend that
     implements the ranges (Triton on Hopper and Blackwell);
  2. while the runner builds a step's attention metadata, each request's `bidi_ranges` are presented to it as one more
     placeholder of its multimodal features, then removed (nothing else, prefix-cache hashing included, sees them);
  3. Gemma 4 clears the ranges on its sliding layers and keeps them on the full-attention layers (the reverse of its
     image rule), which is the mask the model reads with.

The v1 model runner is required (VLLM_USE_V2_MODEL_RUNNER=0): 0.29's default v2 runner never computes the ranges.
The state's range ends before the question, so the questions and option orders of one request share the state's
prefill through prefix caching.
"""
import os

_ACTIVE = os.environ.get("ANYJEV_BIDI_GLOBAL") == "1"


class _Range:
    """The placeholder interface the runner reads: `extract_embeds_range()` -> [(start, end), ...]."""

    def __init__(self, ranges):
        self._ranges = ranges

    def extract_embeds_range(self):
        return list(self._ranges)


class _Feature:
    modality = "anyjev-bidi"

    def __init__(self, ranges):
        self.mm_position = _Range(ranges)


def _ranges_of(req_state):
    sp = getattr(req_state, "sampling_params", None)
    xr = (getattr(sp, "extra_args", None) or {}).get("bidi_ranges") if sp is not None else None
    if not xr:
        return None
    return [(int(xr[i]), int(xr[i + 1])) for i in range(0, len(xr) - 1, 2)]


def _patch_model_config():
    from vllm.config.model import ModelConfig
    ModelConfig.is_mm_prefix_lm = property(lambda self: True)


def _patch_runner():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    orig = GPUModelRunner._build_attention_metadata

    def build(self, *args, **kwargs):
        added = []
        for req_id in self.input_batch.req_ids:
            rs = self.requests.get(req_id)
            rg = _ranges_of(rs) if rs is not None else None
            if rg:
                rs.mm_features.append(_Feature(rg))
                added.append(rs)
        try:
            return orig(self, *args, **kwargs)
        finally:
            for rs in added:
                if rs.mm_features and isinstance(rs.mm_features[-1], _Feature):
                    rs.mm_features.pop()

    GPUModelRunner._build_attention_metadata = build


def _patch_gemma4():
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.models import gemma4_mm

    def clear_sliding(self):
        full = getattr(self, "_full_attn_layer_idxs", None)
        if not full:
            return
        meta = get_forward_context().attn_metadata
        if meta is None:
            return

        def process(d):
            for name, m in d.items():
                if ".layers." not in name:
                    continue
                try:
                    idx = int(name.split(".layers.")[1].split(".")[0])
                except (ValueError, IndexError):
                    continue
                if idx not in full:                           # a sliding layer: causal
                    for attr in ("mm_prefix_range", "mm_prefix_range_tensor", "mm_prefix_query_range_tensor"):
                        if hasattr(m, attr):
                            setattr(m, attr, None)

        for d in (meta if isinstance(meta, list) else [meta]):
            if isinstance(d, dict):
                process(d)

    gemma4_mm.Gemma4ForConditionalGeneration._clear_mm_prefix_for_full_attn_layers = clear_sliding


def register():
    """vllm.general_plugins entry point; a no-op unless ANYJEV_BIDI_GLOBAL=1."""
    if not _ACTIVE:
        return
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")    # for the processes vLLM starts after this one
    _patch_model_config()
    _patch_runner()
    _patch_gemma4()

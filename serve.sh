#!/bin/bash
# The model server the engine reads: AnyJev Tacitus on one GPU, FP8 at load, with the anyjev-vllm-bidi plugin.
#   pip install -r requirements.txt && pip install ./serving
# The model's full-attention layers read each prompt in two bidirectional blocks (the state, and the question with
# its options); the engine sends those token ranges with every request and the plugin applies them.
#   ANYJEV_BIDI_GLOBAL=1           turns the plugin on (it is a no-op otherwise)
#   VLLM_USE_V2_MODEL_RUNNER=0     the ranges are computed by vLLM's v1 model runner (0.29 defaults to v2)
#   TRITON_ATTN                    the attention backend that applies per-request ranges on Hopper and Blackwell GPUs
#   image limit 1                  vLLM keeps its bidirectional-range mode only while multimodal inputs are enabled
#   max-num-batched-tokens         at least the window, so the longest prompt fits in one prefill step
#   prefix caching                 a state's range ends before the question: one prefill of a state serves every
#                                  question and option order of its request
#   processed_logprobs / 32        the label restriction applies before the logprobs (up to 26 letter labels)
PORT=${PORT:-8000}
MAXLEN=${MAXLEN:-32768}       # one 94-96 GB GPU; the longest public prompt is 23,615 tokens. The sliding-window layers
                              # hold KV for a whole prefill chunk, so the window, not the model length, sets the memory
MODEL=${MODEL:-morriszjm/AnyJev-Tacitus-31B}
export ANYJEV_BIDI_GLOBAL=1 VLLM_USE_V2_MODEL_RUNNER=0
vllm serve "$MODEL" --quantization fp8 --logprobs-mode processed_logprobs --max-logprobs 32 --enable-prefix-caching \
  --max-model-len "$MAXLEN" --max-num-batched-tokens "$MAXLEN" --attention-backend TRITON_ATTN \
  --limit-mm-per-prompt '{"image": 1, "video": 0}' --gpu-memory-utilization 0.90 --max-num-seqs 256 \
  --host 127.0.0.1 --port "$PORT"

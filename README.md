# AnyJev Tacitus · Gemma-4-31B

AnyJev Tacitus is a decision model fine-tuned from [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it).
It reads every typed decision from a single prefill, as a probability over the options, without generating tokens.
Its training and its inference were designed together; the serving side ([AnyJev](https://github.com/nokia-applied-research/AnyJev)'s
readout and a small vLLM plugin) is in this repository, and the method will be described in a forthcoming paper.

Model: [morriszjm/AnyJev-Tacitus-31B](https://huggingface.co/morriszjm/AnyJev-Tacitus-31B).

## Decision Index 0.3, public suite

| | |
|---|---|
| public index | **65.52** (raw 73.50, breadth 64.80) |
| areas | knowledge 52.8 · language 70.3 · retrieval 67.5 · tools 81.3 · arts 53.2 |
| requests | 140,178 of 140,178 scoreable, all answered, none unsupported, no errors |
| results | [morriszjm/decision-index-results](https://huggingface.co/datasets/morriszjm/decision-index-results), `runs/anyjev-tacitus-31b-v1` |

## The readout

- Each question is rendered with AnyJev's prompt; the answer is the next-token distribution restricted to the option
  labels, letters up to 26 options and a letter and a digit per option above (A0..A9, B0..B9, ...),
  p(option) = p(letter) · p(digit | letter). Yes/no questions are shown with their criteria. Nothing is generated.
- The options are read in two orders, the one given and its half rotation, and averaged in log space. No label prior.
- The model's full-attention layers read each prompt in two bidirectional blocks: the state, and the question with its
  options. The engine sends the two token ranges with every request and the plugin in `serving/` applies them, so one
  prefill of a state serves every question and option order of its request.

## Run it

```bash
pip install -r requirements.txt                # vLLM 0.29.0, anyjev 0.3.0
pip install ./serving                          # the vLLM plugin (a no-op unless ANYJEV_BIDI_GLOBAL=1)
pip install -e <decision-index checkout>       # the kit, commit 9eb2dbe
bash serve.sh                                  # one GPU, port 8000
bash run.sh                                    # the public 0.3 suite, then the kit's scorer
```

## Settings of the submitted run

| | |
|---|---|
| model | `morriszjm/AnyJev-Tacitus-31B`, bf16 weights quantized to FP8 when vLLM loads them |
| server | vLLM 0.29.0 (torch 2.13.0, CUDA 12.9) with the plugin, the flags of `serve.sh`; window 114,688 tokens |
| engine | `anyjev_engine:AnyJevEngine` from this repository on anyjev 0.3.0: `level=L0 max_rotations=2 big_rotations=2 bidi=block prior=none` |
| kit | apolinario/decision-index @ `9eb2dbe2a358004c8782c66e40a83ac07b953fec`, edition 0.3 |
| hardware | 4× NVIDIA H100 NVL 94 GB, as two tensor-parallel-2 servers; the run was split over group-whole shards, every shard with the settings above |
| declared limits | a prompt longer than the server window, or a choice over 260 options, is refused as unsupported; none occurs in the public suite, whose longest prompt is 23,615 tokens |

`serve.sh` runs one GPU with a 32,768-token window; it reads the same answers as the two-GPU servers of the run, up to
the server's own run-to-run variation, and fits on one RTX PRO 6000.

## Licence

The code in this repository is Apache-2.0. The model's licence is on its model card.

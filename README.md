# Efficient-Content-Moderation-Benchmark-Project

I built this project to explore how we can make AI models that detect hate 
speech smaller, faster, and cheaper to run — without sacrificing accuracy 
utilizing quantization. The techniques I used here were inspired by quantization 
optimization work I did while consulting for Adobe. I continued to research 
how different companies were using quantization and took inspriration from 
Google Gemini's results after using QAT to optimize for runtime and quality.

## What this project does

I took a pretrained language model (DistilBERT) and tested four different 
approaches to running it on a hate speech dataset:

1. **FP32 baseline** — the standard full-size model with no modifications
2. **INT8 dynamic PTQ** — compressing the model after training using 
   8-bit integers instead of 32-bit floats
3. **INT8 QAT** — similar compression, but done in a smarter way that 
   preserves accuracy better
4. **LoRA fine-tuning** — instead of retraining the whole model, I only 
   updated 1.5% of its parameters using adapters

The big question I wanted to answer: which approach gives you the best 
accuracy, smallest size, and fastest inference?

## Results

| Model | Accuracy | F1 Score | Size (MB) | Inference (s) |
|---|---|---|---|---|
| FP32 (32-bit floating point) baseline | 0.505 | 0.497 | 255.4 | 2.02 |
| INT8 (8-bit integer) dynamic PTQ (post-training quantization) | 0.592 | 0.269 | 91.0 | 59.34 |
| INT8 QAT (quantization-aware training) | 0.512 | 0.494 | 255.4 | 62.54 |
| LoRA (low-rank adaptation) fine-tuned | **0.477** | **0.598** | **~67** | **0.67** |

### Why F1 score matters more than accuracy here

The dataset has way more "not hateful" tweets than "hateful" ones, so a 
model can cheat by just guessing "not hateful" every time and still get 
decent accuracy. F1 score catches this — it measures how well the model 
actually finds the hate speech, not just how often it guesses right. That 
makes F1 the metric that actually matters for content moderation.

### What I found

- **LoRA had the best F1 by far (0.598)** and was also the fastest at 
  inference (0.67s) — 3x faster than the FP32 baseline
- **Naive INT8 compression destroyed F1** (0.497 → 0.269) even though 
  accuracy looked okay on the surface — this is a real danger in production 
  systems that I think is underappreciated
- **QAT fixed that problem** — it matched INT8's compression goal while 
  keeping F1 almost identical to the baseline (0.494 vs 0.497), showing 
  that how you quantize matters as much as whether you quantize
- Training only 1.5% of parameters with LoRA was enough to meaningfully 
  improve the model — you don't need to retrain everything

## Larger Context

If you're running content moderation at the scale of a large company — 
billions of posts a day — a model that is 3x faster at inference and 64% 
smaller saves an enormous amount in compute costs. But this project also 
shows that you can't just compress a model blindly. Naive quantization 
cut F1 nearly in half on this dataset, which in a real system would mean 
missing huge amounts of hateful content.

## Tools I used

- `transformers` — loaded and ran DistilBERT
- `peft` — implemented LoRA adapters
- `torchao` — ran quantization-aware training
- `bitsandbytes` — quantization utilities
- `datasets` — loaded the tweet_eval hate speech dataset
- `evaluate` — computed accuracy and F1 scores

## Dataset

[tweet_eval hate](https://huggingface.co/datasets/tweet_eval) — real tweets 
labeled as hateful (1) or not hateful (0).
- Training: 2,400 samples
- Evaluation: 600 samples

## LoRA setup
```python
LoraConfig(
    r=16,            # rank — controls how big the adapters are
    lora_alpha=32,   # scaling factor
    lora_dropout=0.1,
    target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],
)
# 1,034,498 trainable parameters out of 67,989,508 total — just 1.52%
```

## How to run it

1. Open `QLoRA_QAT_Content_Moderation.ipynb` in Google Colab
2. Go to Runtime → Change runtime type → T4 GPU
3. Run all cells from top to bottom
4. Total runtime is about 25 minutes

## Papers I referenced

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [torchao quantization library](https://github.com/pytorch/ao)
- [tweet_eval benchmark](https://huggingface.co/datasets/tweet_eval)
- [Personal research on gemini quantization](https://docs.google.com/document/d/1rIVvR0m340Ki0d5uPSxrqD2RCH2La4BnyfZ817yrizQ/edit?tab=t.0)
- [google research on turboquant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [NVIDIA intro to quantization](https://developer.nvidia.com/blog/model-quantization-concepts-methods-and-why-it-matters/)
- [Gemini technical report](https://arxiv.org/abs/2507.06261)

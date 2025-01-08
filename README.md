# GPT2

I'm making a GPT-2 model from scratch, which I'm building, debugging and training following [Andrej Karpathy's Makemore series](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

To ensure a deep understanding, I typically watch Karpathy's lecture in-depth, then re-implement everything on my own and thoroughly recall the steps. Whenever I get stuck, I reason through from the ground up until I can get to the right answer (rather than just Googling the answer right away).

In parallel, I'm following ETH Zurich's [Natural Language Processing course (from Rycolab)](https://rycolab.io/classes/intro-nlp-f24/), to get additional theoretical background.

## Later plans
After I get a running GPT-2, I want to explore these two changes to the algorithm:
1. Byte-latent transformer (patches instead of the Byte-Pair Encoding tokenizer). [Meta Paper](https://ai.meta.com/research/publications/byte-latent-transformer-patches-scale-better-than-tokens/)
2. Per-token loss tracking (in order to see how much each input sample contributed to the training) & seeing if this can be useful for unlearning. [Inspired by John Carmack](https://x.com/ID_AA_Carmack/status/1874931200697217049)
3. Read through in-depth the Meta Llama 3 Paper [Meta Paper](https://arxiv.org/abs/2407.21783)

### Even later plans
Eventually, it would be cool to build a GPT-2 which was trained on a higher quality, custom dataset. I'm thinking of using e.g. Groq to generate a dataset of conversational/friendly speaking samples (and possibly this leads to a smaller, better-performing model than the naive GPT-2. (Otherwise I'll test out using teacher-student distillation techniques).

If my model performs well, I'll deploy it (perhaps through SageMaker or otherwise) and use it to generate the prompts of my RAG product (https://blog-chatify.vercel.app)

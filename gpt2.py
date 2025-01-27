"""
Fyi this code is quite messy (basically me following along the Karpathy tutorial, & coding it out)

It wouln't make much sense for me to clean this up/document properly because 
if you'd like clean, documented code, you can just check the source: https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import time
import inspect
import os
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from hellaswag import iterate_examples, render_example

# ---------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        # Implementation of the self-attention.
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Scale down by num heads to have result 0 < x < 1
        
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v

        

        y = y.transpose(1, 2).contiguous().view(B, T, C) #Reassemble the heads side by side
        
        y = self.c_proj(y)
        return y



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd) #h layer is 4*fan_in in gpt2 paper
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # During backprop, our x node splits into x and the attention values. 
        # Thus our residual connection directly updates our initial x grads, but also accumulates from attn/mlp layers 
        # This is useful for very deep networks where gradients could be lost otherwise
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), #token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd), #position embedding
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)), #attention + ln + ffwd + ln
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight sharing scheme:
        # Input embeddings (similar tokens should be close) and Output embeddings (similar predictions should be close) have the same purpose
        # So we can actually use the input wte as the output linear layer. This leads to significant speedup.
        # Now wte gets gradients from the classifier layer but also from the entire network.
        # This saves n_embd*token_size = ~38M params saved (around 30% of our 124M model)
        self.transformer.wte.weight = self.lm_head.weight

        # Init params (similar to GPT-2 initialization)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) #by default, pytorch uses uniform
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x) #B, T, vocab_size

        logits = self.lm_head(x) #B, T, vocab_size

        loss = None
        if targets is not None:
            # Adjust view. This ensures that for each batch & time, we compare channel output against expected index
            logits = logits.view(-1, logits.size(-1)) #B*T, C
            targets = targets.view(-1) #B*T

            loss = F.cross_entropy(logits, targets) #B, T, vocab_size & B, T

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600)
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] #discard mask

        # Init hugging face model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn_masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]    

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # Implement the weight decay, only weight decay 2D params (those from matrix mul/embeddings)
        # Weight decay forces network to distribute tasks across all (rather than some weights getting high importance)

        # Get all the parameter groups and the params with gradients
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Weight decay all 2D parametres
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == 'cuda'

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        return optimizer
        

# ----------------------------------------------------------------------------------------------------
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename) #Load the np.uint16 values from the file efficiently
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, world_size, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.world_size = world_size
        assert split in {'train', 'val'}

        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]

        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")

        self.reset()

    def next_batch(self):
        B, T = self.B, self.T

        # TODO: Ideally we want to randomize batching so that our model doesn't get biased by data order
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:].view(B, T))
        
        # Reset the pointer if the next batch would be longer than the tokens length.
        self.current_position += B * T * self.process_rank
        if self.current_position + self.process_rank * B * T + 1 > len(self.tokens):
            # Advance shard/loop, load in new tokens
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank

        return x, y

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# ----------------------------------------------------------------------------------------------------

# Simple launch:
# python gpt2.py

# DDP launch (for eg 8 gpus)
# torchrun --standalone --nproc_per_node=8 gpt2.py


# torchrun command set-up up the RANK, LOCAL_RANK And WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 #is this a ddp run
if ddp:
    assert torch.cuda.is_available(), "we need cuda for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK']) #rank of the gpu globally (across all nodes) <- we use this
    ddp_local_rank = int(os.environ['LOCAL_RANK']) #local rank of the gpu within the current node (we don't care for single node)
    ddp_world_size = int(os.environ['WORLD_SIZE']) # number of processes running
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 #handles logging, checkpointing etc
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    # Attempt to auto-detect device
    device = 'cpu'
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = 'mps' #apple mbp uses metal performance shaders/gpu api
        # NOTE: mps gives really weird bugs (negative losses) which is impossible. CPU does not.
    # device = 'cpu' #OVERRIDE

device_type = "cuda" if device.startswith("cuda") else "cpu"

#Get a data batch
enc = tiktoken.encoding_for_model('gpt2')
# with open('input.txt', 'r') as f:
#     text = f.read()

# text = text[:1000]
# tokens = enc.encode(text)
# B, T = 4, 32

# buf = torch.tensor(tokens[:B*T + 1])
# buf = buf.to(device)
# x = buf[:-1].view(B, T)
# y = buf[1:].view(B, T)

#Set the float32 to use 'high' (Tensor Precision (TP32)) instead of 'highest' (FP32 precision)
torch.set_float32_matmul_precision('high')

# Get logits
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)

use_compile = False #interferes with HellaSwag & evaluation
if use_compile:
    model = torch.compile(model) #fuses nodes. Seems to not work on MBP CPU ;(
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# We can't fit the GPT-2 style batch into one go. 
# Instead, we accumulate gradient across multiple iterations and then optimize when we hit total_batch_size
# Much larger batch sizes results in more stable training, thus grad_accum makes sense
total_batch_size = 524288
B = 64 #16
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f'total desired batch size: {total_batch_size}')
    print(f'=> calculated gradient accumulation steps: {grad_accum_steps}')

# logits, loss = model(x, y)

# print(logits.shape)
# print(loss.item()) #prints 11. -ln(1/50257) = 10.82, which shows our model is not confidently loss. Good starting init.

# Cosine-decay learning rate
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 #total tokens / total_batch_size

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# optimize
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

# Logger
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: #starts empty
    pass


# train_loader = DataLoaderLite(B=4, T=32)
train_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size, split="train") #B=16 does not fit on MPS apple gpu *cries*
val_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size, split="val")

if master_process:
    print("Starting training")

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)
    
    # if step == 0 and master_process:
    #     print("Training loop began")

    # Once in a while sample the validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            # print("Validating model loss...")
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # Might also need to add optimizer state dict and rng seed etc

                torch.save(checkpoint, checkpoint_path)

    # Once in a while evaluate the model on HellaSwag
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                pass
            
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)

            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)

        # Reduce stats across all ranks
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file,"a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")


    # Once in a while sample the model
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:

            with torch.no_grad():
                logits, loss = model(xgen)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)

                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat((xgen, xcol), dim=1)
            
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")


    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

    # if master_process and step < 2:
    #     print(f"Step {step}: Starting {grad_accum_steps} gradient accumulation steps...")

    # if step == 0 and master_process:
    #     print(f"Beginning grad accumulation ({grad_accum_steps} steps)")

    for micro_step in range(grad_accum_steps):
        # if step == 0 and micro_step == 0 and master_process:
        #     print("Training micro_steps began")

        x, y = train_loader.next_batch()
        x = x.to(device)
        y = y.to(device)

        # if step == 0 and micro_step == 0 and master_process:
        #     print(f"Batch loaded, moving to device {device}...")

        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        if torch.cuda.is_available():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16): #adds 5000 tokens/sec on A100 (BF16)
                logits, loss = model(x,y)
        else:
            logits, loss = model(x,y)
        
        loss = loss / grad_accum_steps #need to scale due to grad_accum
        loss_accum += loss.detach()
        
        if ddp:
            # Ensure we only synchronize gpu losses at the end of grad accum (using the ddp sync flag)
            model.require_backward_grad_sync == (micro_step == grad_accum_steps - 1)

        loss.backward()
    
    if ddp:
        # Every node broadcasts the local losses and averages it
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    #calculate global norm. For every grad of every param, square it, add it all, take the sqrt => norm. Ensure it's not more than 1.0
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize() #Wait for gpu to finish all the scheduled tasks
    elif device == 'mps':
        torch.mps.synchronize()
    t1 = time.time()
    dt = (t1 - t0)
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt

    if master_process:
        print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:2f}ms | tok/sec: {tokens_per_sec}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

# print("Finished training")

if ddp:
    destroy_process_group()

import sys;sys.exit(0)

# 
num_return_sequences = 5
max_length = 30


model = GPT.from_pretrained('gpt2') #Run with HF model weights
# model = GPT(GPTConfig()) #Run with uninitialized weights

model.eval()
model.to(device) #model lives on GPU

#prefix tokens
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

torch.manual_seed(42)
# torch.cuda.manual_seed(42)

while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x) #(B, T, vocab_size)

        # take logits at the last position
        logits = logits[:, -1, :] #B, vocab_size

        probs = F.softmax(logits, dim=-1)


        # USing only top 50, it helps ensure we don't take very rare tokens (help keep model from blabbing)

        # Get the top 50
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

        # Select a token from the topk probs
        ix = torch.multinomial(topk_probs, 1) #B, 1

        # Gather corresponding indices sampled from the probs 
        xcol = torch.gather(topk_indices, -1, ix) #B, 1

        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
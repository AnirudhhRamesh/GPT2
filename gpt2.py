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
    ptt = torch.tensot(npt, dtype=torch.long)
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

        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

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
    device = 'cpu' #OVERRIDE

#Get a data batch
# enc = tiktoken.encoding_for_model('gpt2')
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
if torch.cuda.is_available():
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
B = 64
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

# train_loader = DataLoaderLite(B=4, T=32)
train_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size, split="train") #B=16 does not fit on MPS apple gpu *cries*

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x = x.to(device)
        y = y.to(device)

        if torch.cuda.is_available():
            with torch.autocast(device_type=device, dtype=torch.bfloat16): #adds 5000 tokens/sec on A100 (BF16)
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
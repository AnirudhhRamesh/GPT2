# Here we implement the transformer model as a PyTorch python file
import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'mps'
eval_iters = 200
n_embd = 60
n_head = 6
n_layer = 2
dropout = 0.2
# ----------

torch.manual_seed(1337)

# Download the dataset
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
print("Loading dataset..")

with open("../data/input.txt", 'r') as f:
    data = f.read()

# Unique chars
vocab = sorted(list(set(data)))
vocab_size = len(vocab)

# Create mappings from char to ints & back
stoi = {c:i for i, c in enumerate(vocab)}
itos = {i:c for i, c in enumerate(vocab)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda x: ''.join([itos[ix] for ix in x])

#1. Encode the entire dataset as numbers, store as a torch.tensor, create train and test splits
dataset = torch.tensor(encode(data))
train_split = int(0.9*dataset.shape[0])
train_data = dataset[:train_split]
val_data = dataset[train_split:]

print("Encoded dataset.")

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) #pick any int up to the last element - block_size
    
    # We generate a batch with batch_size independent/random rows, each of length block_size
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y

# Estimate the loss by averaging across batch losses (as batch losses are very volatile / lucky or unlucky)
@torch.no_grad()
def estimate_losses():
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size)
        self.query = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x) # (B, T, C)

        # Implementation of attention
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5) #NOTE: Karpathy uses C, but we should actually scale with head_size. 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei) #randomly prevent some nodes from communicating

        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """
    Having multiple heads (each of thus smaller size) allows us to have multiple independent 'communication' channels between the token/nodes
    This could result in focusing on different aspects (e.g. tone, position, consonant/vowel etc). 
    Since the weights are initialized randomly, each will focus on different aspects
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) #just a linear transformation of outcome of the layer
        self.droput = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate the output of multiple heads along the C/embedding dimension. 
        # Since we have multiple heads of smaller head_size, concatenating results in the same size head as the original.
        # Each part of the concatenated head will give us information focused on different parts
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.droput(self.proj(out))
        return out

class FeedForward(nn.Module):
# Simple MLP which gives the model 'time to process the communicated information'
# This is applied 'per token' ie each token/position computes the gather information
   
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout), #Add right before residual. Helps to ensure all neurons are used (otherwise could over-use only few)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # We add in residual connections to help with optimizations.
        # During the backward pass, residual connections allows gradients to fly through the nn (directly through simple addition)
        # Then slowly sa/ffwd start kicking in

        # This helps to prevent e.g. vanishing gradients and thus ensures we can update lower down the network while backpropagating

        x = x + self.sa(self.ln1(x)) #residual connections
        x = x + self.ffwd(self.ln2(x)) #residual connections
        return x

# Implement Bigram class
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size) #Need a linear layer to extract the vocab output from the embeddings

    def forward(self, idx, targets=None):
        B, T = idx.shape

        #embeddings
        tok_emb = self.token_embedding_table(idx) #B, T, C
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # T, C We also want to encode in positional information (the tok_embed alone does not contain this)

        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x) #B, T, vocab_size

        if targets is None:
            loss = None
        else:
            # PyTorch works with B, C, T (instead of B, T, C) thus we need to change the view
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        
        for _ in range(max_new_tokens): #We iterate as many times as tokens we want (unlike before, we don't stop on a '.' token)
            #TODO: Need to crop the context to block size, otherwise our position embedding table will run out of scope
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            #With Bigram model, we only care about the last character
            logits = logits[:, -1, :] #Batch, Time, Channel/Embedding. With bigram, we only select the last context
            probs = torch.softmax(logits, dim=1) #Note: dim=1 because we want a softmax on the embeddings/distribution not the batch/time

            idx_next = torch.multinomial(probs, num_samples=1, replacement=True)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    

model = BigramLanguageModel()
m = model.to(device) #Copy the model weights to the GPU!

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) #Can also use higher lr for smaller networks (e.g. 1e-3)

# Model training
for iter in range(max_iters):

    # Every once in a while, evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_losses()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Load a random sample
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

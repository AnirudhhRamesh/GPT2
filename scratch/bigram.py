import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
# ----------

torch.manual_seed(1337)

# Download the dataset
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", 'r') as f:
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

# Implement Bigram class
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) #embedding output size = vocab size

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) #B, T, C

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
            logits, loss = self(idx)
            #With Bigram model, we only care about the last character
            logits = logits[:, -1, :] #Batch, Time, Channel/Embedding. With bigram, we only select the last context
            probs = torch.softmax(logits, dim=1) #Note: dim=1 because we want a softmax on the embeddings/distribution not the batch/time

            idx_next = torch.multinomial(probs, num_samples=1, replacement=True)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel(vocab_size)
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

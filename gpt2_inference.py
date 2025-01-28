import torch
from gpt2 import GPT, GPTConfig
import tiktoken
import time

def load_model(checkpoint_path, device='cuda'):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device) 
    print("Loaded!")
    
    # Create model with the same config
    model = GPT(checkpoint['config'])
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model'])
    
    # Move model to device and set to eval mode
    model.to(device)
    model.eval()
    
    return model

def generate(model, prompt, max_tokens=100, temperature=1.0, top_k=50, device='cuda'):
    # Encode the prompt
    enc = tiktoken.encoding_for_model('gpt2')
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # Add batch dimension
    tokens = tokens.to(device)
    
    # Generate tokens
    with torch.no_grad():
        while tokens.size(1) < len(tokens[0]) + max_tokens:
            # Get predictions
            logits, _ = model(tokens)
            logits = logits[:, -1, :] / temperature
            
            # Optional: top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample from the distribution
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            print(enc.decode([next_token.item()]), end='', flush=True)

            # Append to sequence
            tokens = torch.cat((tokens, next_token), dim=1)
            
            # Optional: stop if we generate an end token
            if next_token.item() == enc.eot_token:
                break
    
    # Decode the generated tokens
    generated_text = enc.decode(tokens[0].tolist())
    return generated_text

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Load model
# model = load_model('gpt2_log/model_19072.pt', device)
model = load_model('gpt2_log/model_19072.pt', device)

# Generate text
t0 = time.time()
# prompt = "Hello, I'm a language model,"
prompt = input("Enter a prompt: ")
print(prompt, end='')
generated = generate(
    model, 
    prompt,
    max_tokens=50,
    temperature=0.8,  # Lower for more focused/conservative text
    top_k=50,         # Helps avoid rare/unwanted tokens
    device=device
)
t1 = time.time()

print(f"Prompt: {prompt}")
print(f"Generated: {generated}")
print(f"Time taken: {((t1 - t0)*1000):.4f}")
from torch._numpy._dtypes import dtype
from sympy.printing.pytorch import torch
from IPython.utils.py3compat import encode
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

print(torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


data_file_name = "input.txt"
with open(data_file_name, 'r') as file:
    data = file.read()


charecters = sorted(list(set(data)))
stoi = {c: i for i, c in enumerate(charecters)}
itos = {i: c for i, c in enumerate(charecters)}
encode = lambda x: [stoi[c] for c in x]
decode = lambda x: "".join([itos[i] for i in x])
tokenized_base_data = torch.tensor(encode(data))
data_length = len(tokenized_base_data)
limit = int(0.9 * data_length)
train_data = tokenized_base_data[:limit]
val_data = tokenized_base_data[limit:]

######
VOCAB_SIZE = len(charecters)
BLOCK_SIZE = 128
BATCH_SIZE = 64
EMBED_DIM = 256
NUM_HEAD = 2
NUM_BLOCKS = 4
head_size = EMBED_DIM//NUM_HEAD
DROPOUT = 0.2


class Head(nn.Module):
    """ one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.query = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.value = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.head_size = head_size

        self.register_buffer('tril', torch.tril(torch.ones((BLOCK_SIZE, BLOCK_SIZE), dtype=torch.float)))
        self.dropout = nn.Dropout(DROPOUT)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        # print(k.shape, q.shape, v.shape, k.transpose(-2, -1).shape)
        qk = q @ k.transpose(-2, -1)
        qk *= self.head_size**-0.5
        wei = qk.masked_fill(self.tril[:T, :T] == 0.0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(NUM_HEAD)])
        self.WH = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.dropout = nn.Dropout(DROPOUT)
    
    def forward(self, x):
        all_heads = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.WH(all_heads)
        out = self.dropout(out)
        return out


class FeedForwardNetwork(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(DROPOUT),
        )
    
    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mh_head = MultiHeadAttention()
        self.ff_nets = FeedForwardNetwork(EMBED_DIM)
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.ln2 = nn.LayerNorm(EMBED_DIM)
    
    def forward(self, input_embeddings):
        input_embeddings = input_embeddings + self.mh_head(self.ln1(input_embeddings))
        input_embeddings = input_embeddings + self.ff_nets(self.ln2(input_embeddings))
        return input_embeddings



class GPTCharecterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokened_embedding_table = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.decoder_blocks = nn.Sequential(*[DecoderBlock() for _ in range(NUM_BLOCKS)])
        self.ln_final = nn.LayerNorm(EMBED_DIM)
        self.lm_head = nn.Linear(EMBED_DIM, VOCAB_SIZE)
    
    def forward(self, inputs, targets=None):
        B, T = inputs.shape
        pos_idx = torch.arange(T, device=device)
        pos_embed = self.position_embedding_table(pos_idx)
        input_embeddings = self.tokened_embedding_table(inputs) + pos_embed
        input_embeddings = self.decoder_blocks(input_embeddings)
        input_embeddings = self.ln_final(input_embeddings)
        logits = self.lm_head(input_embeddings)

        if targets is None:
            loss = None
        else:
            all_chanels = logits.view(BATCH_SIZE*BLOCK_SIZE, VOCAB_SIZE)
            targets = targets.view(BATCH_SIZE*BLOCK_SIZE)
            loss = F.cross_entropy(all_chanels, targets)
        return logits, loss
    
    def generate(self, inputs, max_length=100):
        window_elements = inputs[:]
        for _ in range(max_length):
            if window_elements.shape[1]>BLOCK_SIZE:
                current_len = window_elements.shape[1]
                window_elements = window_elements[:, current_len-BLOCK_SIZE:]
            logits, _ = self(window_elements) # B, block_size, vocab_length
            last_channels = logits[:, -1, :]
            probs = F.softmax(last_channels, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            inputs = torch.cat((inputs, next_idx), dim=1)
            window_elements = torch.cat((window_elements, next_idx), dim=1)
        return inputs


def get_batch(split="train"):
    target_data = train_data
    if split == "val":
        target_data = val_data
    idx = torch.randint(0, len(target_data)-BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([target_data[i:i+BLOCK_SIZE] for i in idx])
    y = torch.stack([target_data[i+1:i+BLOCK_SIZE+1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y

if __name__ == "__main__":
    model = GPTCharecterModel()
    model = model.to(device = device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    stepi = []
    lossi = []

    for step in range(20000):
        xb, yb = get_batch(split="train")
        model.zero_grad(set_to_none=True)
        logits, loss = model(xb, yb)
        if step % 50 == 0:
            print(f"Step {step}: Loss {loss.item():.4f}")
            stepi.append(step)
            lossi.append(loss.item())
        loss.backward()
        optimizer.step()
    print(f"Final Training Loss:- {loss.item()}")


    plt.plot(stepi, lossi)
    plt.title("Training Loss")
    plt.xlabel("Step (x50)")
    plt.ylabel("Loss")
    plt.savefig('loss_curve.png')
    print("Loss curve saved to loss_curve.png")


    torch.save(model.state_dict(), "char-model.pth")
    print("Model weights saved to char-model.pth")


    print("\nGenerating text...")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    context[0,0] = stoi['T'] if 'T' in stoi else 0

    generated_ids = model.generate(context, max_length=1000)[0].tolist()
    generated_text = decode(generated_ids)

    output_filename = "more_output.txt"
    with open(output_filename, "w", encoding='utf-8') as f:
        f.write(generated_text)

    print(f"Generated text saved to {output_filename}")
    print("\nSample preview:")
    print(generated_text[:200])

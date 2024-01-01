
class SelfAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        self.k_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = torch.nn.Dropout(dropout)

        self.softmax = torch.nn.Softmax(dim = -1)

    def forward(self, x1, x2):
        k = self.k_proj(x1)
        v = self.v_proj(x2)
        q = self.q_proj(x2)

        q = q.reshape(q.shape[0], q.shape[1], self.num_heads, q.shape[-1] // self.num_heads).transpose(2, 1)
        k = k.reshape(k.shape[0], k.shape[1], self.num_heads, k.shape[-1] // self.num_heads).transpose(2, 1)
        v = v.reshape(v.shape[0], v.shape[1], self.num_heads, v.shape[-1] // self.num_heads).transpose(2, 1)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.k_proj.weight.shape[-1])
        attn = attn + torch.ones_like(attn) * (-1e5)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(2, 1).reshape(attn.shape[0], attn.shape[1], -1)
        out = self.out_proj(out)
        return out

# Initializing the model
m = SelfAttention(256, 8)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)

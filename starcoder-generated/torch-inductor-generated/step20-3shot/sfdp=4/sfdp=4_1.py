 
class MultiheadAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)
 
    def forward(self, k, v, q, attn_mask=None):
        bs, l = k.size(0), k.size(1)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = self.q_proj(q)
        q = q * self.scaling
        k = k.view(bs, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, l, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.view(bs, l, self.num_heads, self.head_dim).transpose(1, 2)
        attn_score = torch.matmul(q, k.transpose(-2, -1))
        attn_score = attn_score.masked_fill(attn_mask, float("-inf"))
        attn_weight = torch.softmax(attn_score, dim=-1)
        attn_weight = F.dropout(attn_weight, p=self.dropout, training=self.training)
        attn_out = torch.matmul(attn_weight, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bs, l, self.embed_dim)
        attn_out = self.out_proj(attn_out)
        return attn_out

# Initializaing the model
model = MultiheadAttention(embed_dim=64, num_heads=4)
print(model)

# Inputs to the model
k = torch.randn(1, 64, 64)  # (batch_size, length, embed_dim)
v = torch.randn(1, 64, 64)  # (batch_size, length, embed_dim)
q = torch.randn(1, 64, 64)  # (batch_size, length, embed_dim)
attn_mask = F.pad(torch.zeros(64, 64), (1, 0), value=1)  # (batch_size, length, 1)
output = model(k, v, q, attn_mask)


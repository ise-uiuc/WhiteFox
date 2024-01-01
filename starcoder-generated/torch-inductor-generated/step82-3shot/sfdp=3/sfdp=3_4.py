
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0.0):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x, attn_mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
 
        q *= float(q.size(-1)) ** -0.5
        q = torch.masked_fill(q, attn_mask.bool(), -1e9)
        attn = torch.softmax(q.matmul(k.transpose(-2, -1)), -1)
        attn = self.dropout(attn)
        attn = attn.unsqueeze(-1).bmm(v).squeeze(-1)
 
        return self.out_proj(attn)

# Initializing the model with an embedding dimension of 8 and a dropout rate of 0.3
embed = 8
dropout = 0.3
m = SelfAttention(embed_dim=8, dropout=dropout)

# Inputs to the model
x = torch.randn(1, 32, 8)
attn_mask = torch.full((32, 32), float("-inf")).tril_(-1)

m(x, attn_mask=attn_mask)



class Model(torch.nn.Module):
    def __init__(self, n_heads, embed_dim, dropout=0.0, attn_dropout=0.0, bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.dropout = dropout
  
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // n_heads
         
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        if self.dropout > 0.0:
            self.attn_dropout_layer = nn.Dropout(attn_dropout)
 
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        if self.dropout > 0.0:
            self.proj_dropout_layer = nn.Dropout(attn_dropout)
 
    def forward(self, x1, x2, attn_mask=None, mems=None):
        device = x1.device
        bsz, qlen, klen, _ = x1.shape
 
        query, key, value = torch.chunk(self.qkv_proj(x1), 3, dim=-1)
        query = query.view(bsz, qlen, self.n_heads, self.head_dim).transpose(1, 2) # [B, n_heads, qlen, head_dim]
        key = key.view(bsz, klen, self.n_heads, self.head_dim).transpose(1, 2) # [B, n_heads, klen, head_dim]
        value = value.view(bsz, klen, self.n_heads, self.head_dim).transpose(1, 2) # [B, n_heads, klen, head_dim]
 
        attn_scores = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        attn_scores = attn_scores + attn_mask
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = attn_weights.matmul(value)
        context = context.transpose(1, 2).reshape(bsz, qlen, embed_dim)
        output = self.proj(context)
        if self.dropout > 0.0:
            output = self.proj_dropout_layer(context)
        return output

# Initializing the model
n_heads = 16
embed_dim = 512
m = Model(n_heads, embed_dim, dropout=0.0, attn_dropout=0.0, bias=True)

# Inputs to the model
x1 = torch.randn(1, 20, 512)
x2 = torch.randn(1, 20, 512)
attn_mask = torch.eye(x1.shape[1]).bool().to("cuda") # [qlen, klen]
attn_mask = attn_mask.unsqueeze(0).expand(x1.shape[0], 20, 20) # [B, qlen, klen]

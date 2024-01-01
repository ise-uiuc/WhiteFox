
class Model(torch.nn.Module):
    def __init__(self, n_head, dim_q, dim_k, dim_v, dim_o, dropout, attn_mask):
        super().__init__()
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_o = dim_o
        self.dim_total = dim_q + dim_k + dim_v
        
        self.w_qkv = torch.nn.Linear(dim_q, n_head * self.dim_total, bias=False)
        self.proj = torch.nn.Linear(n_head * self.dim_total, dim_o, bias=False)
        self.attn_mask = attn_mask
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, q, k, v):
        qkv = self.w_qkv(q)
        qs = qkv[:, :, :self.dim_q].contiguous().view(q.size(0), -1, self.dim_q)
        ks = qkv[:, :, self.dim_q:self.dim_k + self.dim_q].contiguous().view(k.size(0), -1, self.dim_k)
        vs = qkv[:, :, self.dim_k + self.dim_q:].contiguous().view(v.size(0), -1, self.dim_v)
    
        at = (qs @ ks.transpose(-2, -1)) / math.sqrt(self.dim_q)
        at = at + self.attn_mask

        at = at / at.sum(-1, keepdim=True).clamp(min=0)
        at = at.masked_fill(self.attn_mask.to(torch.bool), -1e9)
    
        v_out = (at @ vs)#.view(at.size(0), at.size(1), -1)
        v_out = v_out.view(v_out.size(0), -1)
        return self.proj(self.dropout(v_out))

# Initializing the model
model = Model(n_head=1, # Number of attention heads
              dim_q=512, # Dimension of hidden queries
              dim_k=64, # Dimension of hidden keys
              dim_v=64, # Dimension of hidden values
              dim_o=512, # Dimension of output
              dropout=0.1,
              attn_mask=torch.randn(1,100,384).unsqueeze(1)) # attention mask (0 to ignore / 1 to attend)

# Inputs to the model

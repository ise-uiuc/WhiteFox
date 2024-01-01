
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, _ = x.size()

        xq = self.query_proj(x).view(B, T, self.n_heads, self.d_head).transpose(2, 1) # B, nh, T, hs
        xk = self.key_proj(x).view(B, T, self.n_heads, self.d_head).transpose(2, 1) # B, nh, T, hs
        v = self.value_proj(x).view(B, T, self.n_heads, self.d_head).transpose(2, 1) # B, nh, T, hs

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head) # B, nh, T, T
        # optionally apply mask
        if mask is not None:
            scores += mask
        attn = torch.softmax(scores, dim=-1) # B, nh, T, T
        context = attn @ v # B, nh, T, hs

        context = context.transpose(2, 1).contiguous().view(B, T, self.n_heads * self.d_head) # B, T, nh*hs

        return self.out_proj(context)

# Initializing the model
m = MultiHeadSelfAttention(d_model=512, n_heads=8)

# Inputs to the model
x = torch.randn(1, 60, 512)
mask = torch.full((1, 60, 60), float('-inf'))

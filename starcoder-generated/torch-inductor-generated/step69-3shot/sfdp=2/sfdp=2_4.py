
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5
        self.multi_head_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
 
    def forward(self, x1):
        v1 = x1.transpose(0,1)
        v2 = self.multi_head_attn(v1, v1, v1)
        v3 = v2.transpose(0,1)
        return v3

# Initializin the model with the default embedding dim, 2048, and the number of heads, 16, and the dropout rate, 0.
m = SelfAttention()

# Inputs to the model
x1 = torch.randn(1,64, 2048)


class Model(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.norm = torch.nn.LayerNorm(embed_dim)
        self.attn = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
 
    def forward(self, x1, x2):
        x1 = self.norm(x1)
        x1, _ = self.attn(query=x1, key=x2, value=x2)
        return x1

# Initializing the model
m = Model(embed_dim=256, num_heads=2)

# Inputs to the model
x1 = torch.randn(2, 5, 256)
x2 = torch.randn(2, 8, 256)

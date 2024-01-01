
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim=8, num_heads=2, dropout=0.2)
 
    def forward(self, x1):
        output, attention = self.attn(x1, x1, x1, attn_mask=None, key_padding_mask=None)[0]
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(27, 8, 64)

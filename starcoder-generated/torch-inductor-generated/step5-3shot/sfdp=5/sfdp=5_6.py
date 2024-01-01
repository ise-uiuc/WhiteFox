
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.__unfused_attn = torch.nn.MultiheadAttention(embed_dim, num_heads)
 
    def forward(self, x1, x2, __mask__):
        v1, v2 = self.__unfused_attn(x1, x2, x2, key_padding_mask=__mask__)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
x2 = torch.randn(5, 8, 25)
mask = torch.zeros(5, 5)

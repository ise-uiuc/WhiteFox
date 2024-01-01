
class Model(torch.nn.Module):
    def __init__(self,
                 hidden_size,
                 num_heads,
                 dropout_p):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(hidden_size, num_heads, dropout_p)

    def forward(self, x1, x2):
        v1 = self.attn(x1, x1, x2)
        return v1

# Initializing the model
m = Model(2, 1, 0.0)

# Inputs to the model
x1 = torch.randn(1, 1, 2)
x2 = torch.randn(1, 1, 2)

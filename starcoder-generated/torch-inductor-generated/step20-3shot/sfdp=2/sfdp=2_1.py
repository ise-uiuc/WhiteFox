
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(n_head=16, d_model=64, dropout=0.1)
 
    def forward(self, x1, x2):
        return self.attention(x1, x1, x2)[0], self.key(x2), self.value(x2)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 3, 64, 64)
x2 = torch.randn(1, 3, 4, 64, 64)
__output__, __key__, __value__ = m(x1, x2)


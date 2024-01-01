
class Model(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(hidden_size, 8)
 
    def forward(self, x1, x2, x3):
        v1 = self.attention(x1, x2, x3)[0]
        return v1

# Initializing the model
m = Model(hidden_size)

# Inputs to the model
x1 = torch.randn(1, 16, 128)
x2 = torch.randn(1, 16, 64)
x3 = torch.randn(1, 16, 64)

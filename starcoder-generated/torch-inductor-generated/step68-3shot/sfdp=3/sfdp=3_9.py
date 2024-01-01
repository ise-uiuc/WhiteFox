
from torch.nn.modules.activation import MultiheadAttention

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = MultiheadAttention(embed_dim=32, num_heads=1)
 
    def forward(self, x1):
        x2, _ = self.attention(x1, x1, x1)
        return x2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, 10, 20)

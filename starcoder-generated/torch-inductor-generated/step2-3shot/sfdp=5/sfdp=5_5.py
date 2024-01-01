
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(64, 4)
 
    def forward(self, x1, x2):
        x3, x4 = self.attn(x1, x2, x2)
        return x3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 4, 28)
x2 = torch.randn(1, 64, 28, 28)

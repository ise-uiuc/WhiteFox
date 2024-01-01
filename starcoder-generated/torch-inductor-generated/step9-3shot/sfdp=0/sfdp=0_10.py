
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 1.0 / math.sqrt(64)
        
    def forward(self, x1, x2, x3):
        w1 = torch.matmul(x1, x2.transpose(-2, -1))
        w2 = w1 * self.scale
        w3 = w2.softmax(dim=-1)
        return torch.matmul(w3, x3)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 4, 1)
x2 = torch.randn(1, 4, 64, 1)
x3 = torch.randn(1, 4, 1, 1)

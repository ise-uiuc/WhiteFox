
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lr = torch.nn.Linear(16, 16, bias=False)
 
    def forward(self, x1):
        v1 = self.lr(x1)
        v2 = v1 + 3
        v3 = x1.clamp_min(0)
        v4 = x1.clamp_max(6)
        v5 = v4 / 6
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16, 28, 28)

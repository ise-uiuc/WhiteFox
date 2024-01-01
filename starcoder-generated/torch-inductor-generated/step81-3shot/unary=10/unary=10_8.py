
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 10)
 
    def forward(self, x3):
        v1 = self.linear(x3)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 * 0.16666666666666666
        return v5

# Initializing the model
m2 = Model()

# Inputs to the model
x3 = torch.randn(5, 64)

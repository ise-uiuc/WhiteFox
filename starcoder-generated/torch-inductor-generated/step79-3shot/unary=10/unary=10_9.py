
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(512, 512)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(224, 224)
x2 = torch.randn(3, 32, 32)
x3 = torch.randn(3, 32, 10, 10)
x4 = torch.randn(3, 32, 32, 32)

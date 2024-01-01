
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(252, 1024)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * torch.clamp(torch.add(v1, 3), min=0, max=6)
        v3 = v2 * 0.16666666666666666
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 252)

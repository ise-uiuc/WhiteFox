
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x2):
        v2 = self.linear(x2)
        v3 = v2 * torch.clamp(v2 + 3, min=0, max=6)
        v4 = v3 / 6
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 3)

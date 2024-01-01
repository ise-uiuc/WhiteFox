
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(l1):
        v1 = self.linear(l1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0, max=6)
        v4 = v3 / 6
        return v4

# Initializing the model
v = Model()

# Inputs to the model
l1 = torch.randn(1, 8)

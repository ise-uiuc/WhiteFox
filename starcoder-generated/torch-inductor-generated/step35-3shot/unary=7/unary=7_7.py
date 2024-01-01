
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 4)
 
    def forward(self, inp):
        v1 = self.linear(inp)
        v2 = torch.clamp(torch.clamp(v1 + 3, min=0), 6)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 32)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 64)
 
    def forward(self, x1):
        o = self.linear(x1)
        o = o + 3
        o = torch.clamp(o, min=0, max=6)
        o = o / 6
        return o

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)

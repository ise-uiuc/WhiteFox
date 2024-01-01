
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(12, 12)
 
    def forward(self, x1):
        r1 = self.linear(x1)
        a1 = torch.clamp(r1 + 3.0, min=0.0, max=6.0)
        v1 = a1 / 6.0
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 12)

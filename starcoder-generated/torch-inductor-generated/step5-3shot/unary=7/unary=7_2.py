
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1, bias=True)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp(torch.add(v1, 3.0), min=0.0, max=6.0)
        v3 = v2 / 6.0
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 1, 3)

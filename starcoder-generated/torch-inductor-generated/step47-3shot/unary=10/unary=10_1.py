
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + 3.0
        v3 = v2.clamp(min=0.0, max=6.0)
        v4 = v3 / 6.0
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)

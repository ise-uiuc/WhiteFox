
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.line = torch.nn.Linear(5, 5)
 
    def forward(self, x1):
        v1 = self.line(x1)
        v2 = v1 + 3
        v3 = v2.clamp(min=0)
        v4 = v2 * v3
        v5 = v4 / 6
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)

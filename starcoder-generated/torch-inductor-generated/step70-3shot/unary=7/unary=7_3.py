
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8, bias=False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1.clamp(0, 6)
        v3 = v2 + 3
        v4 = v3 / 6
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)

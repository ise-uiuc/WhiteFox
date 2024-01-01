
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, N)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = (v1 + 3).clamp(0, 6)
        v3 = v2 / 6
        return v1 + v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

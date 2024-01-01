
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        a1 = v1 + 3
        v2 = torch.clamp(a1, 0, 6)
        v3 = v2 * 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)

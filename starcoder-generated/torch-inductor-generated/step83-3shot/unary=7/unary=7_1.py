 2
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(20, 20)
 
    def forward(self, x1):
        v2 = self.linear(x1)
        v3 = v2 * torch.clamp(v2 + 3, 0, 6)
        v4 = v3 / 6
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 20)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 6)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 * torch.clamp(v1 + 3, min=-3, max=3)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(3, 3)

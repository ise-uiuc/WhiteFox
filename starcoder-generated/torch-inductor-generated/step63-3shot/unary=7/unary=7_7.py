
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 * torch.clamp(torch.clamp(v1 + 3, min=0), max=6)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 5)

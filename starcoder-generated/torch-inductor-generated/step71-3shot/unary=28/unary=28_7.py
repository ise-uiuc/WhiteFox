
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)
 
    def forward(self, x1):
        v1 = torch.clamp_min(self.linear(x1), min=2)
        v2 = torch.clamp_max(v1, max=3)
        return torch.tanh(v2)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)

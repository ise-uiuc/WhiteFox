
class Model(torch.nn.Module):
    def __init__(self, min_value=2):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
 
    def forward(self, x):
        v = self.linear(x)
        return torch.clamp_min(torch.clamp_max(v, min=2), max=4)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(5, 1)

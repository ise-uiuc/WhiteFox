
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
 
    def forward(self, x):
        v1 = self.linear(x)
        return torch.clamp_max(torch.clamp_min(v1, min_value=10), max_value=30)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 2)

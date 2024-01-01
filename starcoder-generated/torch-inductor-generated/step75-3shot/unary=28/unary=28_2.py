
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 16)
 
    def forward(self, x1):
        return torch.clamp_max(torch.clamp_min(self.linear(x1), max_value=50), min_value=0)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)

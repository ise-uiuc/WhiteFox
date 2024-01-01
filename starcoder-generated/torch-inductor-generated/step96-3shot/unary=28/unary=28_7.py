
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x1):
        return torch.clamp_max(torch.clamp_min(self.linear(x1), -0.4), 0.01)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)

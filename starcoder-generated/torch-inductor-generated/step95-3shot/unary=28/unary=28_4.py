
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
 
    def forward(self, x):
        return torch.clamp_max(torch.clamp_min(self.linear(x), 1.3), 0.5)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 2)

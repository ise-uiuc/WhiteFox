
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(136, 90)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min=-1.0)
        return torch.clamp_max(v2, max=0.5)

# Initializing the model
m = Model()

minValue = torch.tensor([-1.0])
maxValue = torch.tensor([0.5])

# Inputs to the model
x1 = torch.randn(1, 136)

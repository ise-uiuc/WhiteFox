
class Model(torch.nn.Module):
    def __init__(self, min_value=1.0, max_value=2.0):
        super().__init__()
        self.linear = torch.nn.Linear(5, 5)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min=self.min_value)
        return torch.clamp_max(v2, max=self.max_value)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)

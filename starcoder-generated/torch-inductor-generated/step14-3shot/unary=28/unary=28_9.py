
class Model(torch.nn.Module):
    def __init__(self, min_value=5, max_value=10):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(1))
        self.linear = torch.nn.Linear(1, 1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.param)
        v3 = torch.clamp_max(v2, self.param)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1)

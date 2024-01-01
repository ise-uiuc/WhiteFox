
class Model(torch.nn.Module):
    def __init__(self, min_value=1.1, max_value=1.9):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.min = min_value
        self.max = max_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min=self.min)
        v3 = torch.clamp_max(v2, max=self.max)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1)

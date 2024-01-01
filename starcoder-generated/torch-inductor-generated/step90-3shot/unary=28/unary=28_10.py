
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x, min_value=-3, max_value=-1):
        v1 = self.linear(x)
        v6 = torch.clamp_min(v1, min_value)
        v7 = torch.clamp_max(v6, max_value)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3)

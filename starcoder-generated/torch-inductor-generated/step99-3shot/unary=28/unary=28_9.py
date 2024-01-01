
class Model(torch.nn.Module):
    def __init__(self, min_value=[-10], max_value=[10]):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value=-10)
        v3 = torch.clamp_max(v2, max_value=10)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 5)

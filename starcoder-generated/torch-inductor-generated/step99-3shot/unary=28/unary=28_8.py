
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        return torch.clamp_min(torch.clamp_max(v1, min_value=0.5), max_value=1.0)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 10)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 5)
 
    def forward(self, x1, min_value=-6.1, max_value=6.1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, float(min_value))
        v3 = torch.clamp_max(v2, float(max_value))
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 6)


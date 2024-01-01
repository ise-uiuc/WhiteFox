
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(23, 42)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value=-3)
        v3 = torch.clamp_max(v2, max_value=3)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 23)


class Model(torch.nn.Module):
    def __init__(self, min_value=0.1, max_value=15.0):
        super().__init__()
        self.linear = torch.nn.Linear(3, 16)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value=self.min_value)
        v3 = torch.clamp_max(v2, max_value=self.max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)

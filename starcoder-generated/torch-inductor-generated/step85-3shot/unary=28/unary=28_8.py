
class Model(torch.nn.Module):
    def __init__(self, min_value=0, max_value=1):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        flat = x1.view(x1.shape[0], -1)
        v1 = self.linear(flat)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)

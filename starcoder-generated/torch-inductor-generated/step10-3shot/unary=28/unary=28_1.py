
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x1):
        v1 = torch.clamp_min(x1, min_value)
        v2 = torch.clamp_max(v1, max_value)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
min_value = torch.randn(1, 1)
max_value = torch.randn(1, 1)
x1 = torch.randn(1, 8)

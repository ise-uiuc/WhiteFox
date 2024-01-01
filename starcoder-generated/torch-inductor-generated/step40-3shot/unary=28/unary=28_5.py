
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(10, 1)
 
    def forward(self, x1, min_value=0, max_value=6.0):
        v1 = self.lin(x1)
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
min_value = 0
max_value = 6.0
x1 = torch.randn(1, 10)

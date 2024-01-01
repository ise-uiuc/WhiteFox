
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm1d(10)
        self.bn2 = torch.nn.BatchNorm1d(10)
 
    def forward(self, x1, min_value, max_value):
        v1 = self.bn1(x1)
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        v4 = self.bn2(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(20, 10)

# Keyword arguments for the model.
min_value = -2
max_value = 2

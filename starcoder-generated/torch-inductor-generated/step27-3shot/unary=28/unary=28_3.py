
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.clamp_min(v1, -12.0)
        v3 = torch.clamp_max(v2, 12.0)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8, 32, 32)
min_value = torch.tensor(-12.0)
max_value = torch.tensor(12.0)

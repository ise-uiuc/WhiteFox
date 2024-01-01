
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = (torch.clamp_min(v1, 0.5) * 2) - 1
        v3 = torch.clamp_max(torch.sign(v2), min=0)
        v4 = v2 * v3 + v3
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
min_value = 0.5
max_value = 1.5

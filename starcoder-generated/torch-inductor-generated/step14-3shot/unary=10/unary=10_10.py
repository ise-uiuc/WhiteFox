
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 16, bias=False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + 3
        v3 = torch.nn.functional.relu(v2)
        v4 = torch.clamp_min(v3, 0)
        v5 = torch.clamp_max(v4, 6)
        v6 = v5 / 6
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

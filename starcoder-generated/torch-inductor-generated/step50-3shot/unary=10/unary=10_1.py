
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 64)
 
    def forward(self, x1):
        b1 = torch.rand(10, 16)
        v2 = self.linear(x1)
        v3 = v2 + 3
        v4 = torch.clamp_min(v3, 0)
        v5 = torch.clamp_max(v4, 6)
        v6 = v5 / 6
        v7 = b1 * (v6)
        return v7

# Initializing the model
m = Model()
torch.manual_seed(0)

# Inputs to the model
x1 = torch.rand(10, 16)

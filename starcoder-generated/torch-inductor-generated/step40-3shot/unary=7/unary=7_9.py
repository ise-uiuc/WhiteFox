
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=5, out_features=8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * torch.clamp_min(torch.clamp_max(v1 + 3, max=6), min=0)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(6, 5)

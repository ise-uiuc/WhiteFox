
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64 * 64, 8)
 
    def forward(self, x1):
        v1 = x1.view(x1.size(0), -1)
        v2 = self.linear(v1)
        v3 = v2 + 3
        v4 = torch.clamp_max(v3, 6)
        v5 = torch.clamp_min(v4, 0)
        v6 = v5 / 6
        return v6
# Initializing the model
m = Model()

# Inputs to the model

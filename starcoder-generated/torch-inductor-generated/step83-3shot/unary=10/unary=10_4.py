
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 8)
 
    def forward(self, x2):
        v3 = self.linear1(x2)
        v4 = v3 + 3
        v5 = torch.clamp_min(v4, 0)
        v6 = torch.clamp_max(v5, 6)
        v7 = v6 / 6
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 8)

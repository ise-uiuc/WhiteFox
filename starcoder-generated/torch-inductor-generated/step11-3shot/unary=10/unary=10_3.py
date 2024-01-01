
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)
 
    def forward(self, x2):
        v4 = self.linear(x2)
        v5 = v4 + 3
        v7 = torch.clamp_min(v5, 0)
        v9 = torch.clamp_max(v7, 6)
        v10 = v9 / 6
        return v10

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 3)

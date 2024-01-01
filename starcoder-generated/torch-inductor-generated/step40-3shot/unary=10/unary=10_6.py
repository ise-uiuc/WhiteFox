
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)
 
    def forward(self, x1):
        v7 = self.linear(x1)
        v8 = v7 + 3
        v9 = torch.nn.functional.clamp_min(v8, 0)
        v10 = torch.nn.functional.clamp_max(v9, 6)
        v11 = v10 / 6
        return v11

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

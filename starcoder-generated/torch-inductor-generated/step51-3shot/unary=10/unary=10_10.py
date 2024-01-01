
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 12)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        v6 = self.linear(x2)
        v7 = v6 + 3
        v8 = torch.clamp_min(v7, 0)
        v9 = torch.clamp_max(v8, 6)
        v10 = v9 / 6
        v11 = torch.cat([v5, v10], axis=1)
        return v11

# Initializing the model
m = Model()

# Inputs to the model (x1, x2)
x1 = torch.randn(1, 4) # the input for the first linear layer
x2 = torch.randn(1, 4) # the input for the second linear layer

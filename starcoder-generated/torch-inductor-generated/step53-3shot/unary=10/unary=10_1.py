
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=3, out_features=1)
     
    def forward(self, x2):
        v1 = self.linear(x2)
        v2 = v1.add(3)
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4/6
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 3)

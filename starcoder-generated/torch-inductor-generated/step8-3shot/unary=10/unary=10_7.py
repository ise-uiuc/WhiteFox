
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(20, 40)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + 3.0
        v3 = v2.clamp_min(0)
        v4 = v3.clamp_max(6.0)
        v5 = v4 / 6.0
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 20)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)
 
    def forward(self, x1, min, max):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min)
        v3 = torch.clamp_max(v2, max)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
k1 = torch.randint(2, 7, (1,), dtype=torch.float32)
k2 = torch.randint(8, 13, (1,), dtype=torch.float32)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 64)
 
    def forward(self, x1):
        t1 = self.linear(x1)
        v1 = t1 + 3
        t2 = torch.clamp_min(v1, 0)
        t3 = torch.clamp_max(t2, 6)
        v2 = t3 / 6
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)

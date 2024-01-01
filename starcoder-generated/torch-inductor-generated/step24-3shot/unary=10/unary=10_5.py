
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 9)
 
    def forward(self, x1):
        r1 = self.linear(x1)
        r2 = r1 + 3
        r3 = torch.clamp_min(r2, 0)
        r4 = torch.clamp_max(r3, 6)
        r5 = r4 / 6
        return r5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

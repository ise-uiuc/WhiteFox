
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(24, 32)
        self.l2 = torch.nn.Linear(32, 64)
        self.l3 = torch.nn.Linear(64, 16)
 
    def forward(self, x1, x2, x3):
        v1 = self.l1(x1)

        v2 = self.l2(v1)

        h1 = torch.cat([x2, v2], dim=1)
        h2 = self.l3(h1)
        return h2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 24)
x2 = torch.randn(1, 8)
x3 = torch.randn(1, 8)

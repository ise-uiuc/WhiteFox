
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(6, 1)
        self.k = torch.nn.Linear(6, 1)
        self.v = torch.nn.Linear(6, 1)
 
    def forward(self, x, mask):
        v1 = self.q(x)
        v2 = self.k(x)
        v3 = self.v(x)
        print(v1.size())
        print(v2.size())
        print(v3.size())
        v4 = v1 @ v2.transpose(-2, -1) / math.sqrt(6)
        v5 = v4 + mask
        v6 = v5 / v5.sum(-1, keepdim=True)
        v7 = v6 @ v3
        return v7

# Initializing the model
q = torch.nn.Linear(6, 1)
k = torch.nn.Linear(6, 1)
v = torch.nn.Linear(6, 1)
m = Model()

# Inputs to the model
x = torch.randn(1, 6)
mask = -10000.0 * (x == 0.0).float()

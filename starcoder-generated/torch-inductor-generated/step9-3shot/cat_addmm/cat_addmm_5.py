
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(8, 16)
        self.lin2 = torch.nn.Linear(16, 32)
        self.lin3 = torch.nn.Linear(32, 64)
 
    def forward(self, x3):
        v1 = self.lin1(x3)
        v2 = self.lin2(v1)
        v3 = torch.addmm(x3, v2, m.mat3)
        v4 = torch.cat((v3), 0)
        v5 = self.lin3(v4)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 8)
x2 = torch.randn(16, 16)
x3 = torch.randn(64, 64)
m.mat3 = torch.randn(16, 16)

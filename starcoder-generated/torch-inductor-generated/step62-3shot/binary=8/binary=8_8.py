
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(32 * 32 * 3, 1024)
        self.l2 = torch.nn.Linear(1024, 1024)
        self.l3 = torch.nn.Linear(1024, 1024)
        self.l4 = torch.nn.Linear(1024, 1024)
    def forward(self, x1):
        v1 = x1.view(-1)
        v2 = self.l1(v1)
        v3 = v2.view(-1, 32, 32, 32)
        v4 = self.l3(v2)
        v5 = self.l4(v3)
        v6 = self.l4(v4)
        v7 = self.l1(v5)
        v8 = self.l3(v6)
        v9 = self.l1(v7)
        v10 = v8 + v9
        v11 = self.l2(v10).pow(2)
        v12 = v11 + 1
        v13 = v11.div(v12)
        return v13
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)

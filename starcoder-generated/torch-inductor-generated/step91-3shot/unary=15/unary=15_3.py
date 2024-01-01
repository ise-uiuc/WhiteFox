
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = torch.nn.Linear(7, 5, bias=True)
        self.linear1 = torch.nn.Linear(5, 32, bias=True)
        self.linear2 = torch.nn.Linear(32, 55, bias=True)
        self.linear3 = torch.nn.Linear(55, 120, bias=True)
        self.linear4 = torch.nn.Linear(120, 28, bias=True)
    def forward(self, x0):
            v0 = x0.permute(0, 2, 3, 1)
            v2 = self.linear0(v0)
            v1 = x0.permute(0, 2, 3, 1)
            v3 = self.linear1(v1)
            v5 = self.linear2(v3)
            v4 = x0.permute(0, 2, 3, 1)
            v6 = self.linear3(v4)
            v8 = self.linear4(v6)
            return v8
# Inputs to the model
x0 = torch.randn(1, 7, 9, 9)

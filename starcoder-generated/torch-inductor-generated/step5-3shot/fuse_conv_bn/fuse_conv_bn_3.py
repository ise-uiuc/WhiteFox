
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = torch.nn.Linear(4, 6, bias=False)
        torch.manual_seed(3)
        self.f1.weight = torch.nn.Parameter(torch.zeros(self.f1.weight.shape[0], self.f1.weight.shape[1]))

        bn = torch.nn.BatchNorm1d(6)
        bn.running_mean = torch.arange(6, dtype=torch.float)
        bn.running_var = torch.arange(6, dtype=torch.float) * 2 + 1

        self.f2 = torch.nn.Sequential(self.f1, bn)
    def forward(self, x):
        x1 = self.f1(x)
        x2 = x1[0:6]
        x3 = x2.mean()
        p = torch.mul(x3, 100)
        return p
# Inputs to the model
x = torch.randn(64)

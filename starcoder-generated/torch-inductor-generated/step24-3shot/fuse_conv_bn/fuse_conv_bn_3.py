
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features=2, out_features=3, bias=False)
        self.linear2 = torch.nn.Linear(in_features=3, out_features=2)
        self.bn = torch.nn.BatchNorm1d(2)
    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x1)
        x3 = self.bn(x2)
        return x2, x3
# Inputs to the model
x = torch.randn(2, 2)

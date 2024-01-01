
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.nn.init.kaiming_normal_(self.lin.weight, nonlinearity="relu")
        self.drop = torch.nn.Dropout(0.5)
    def forward(self, x1):
        v1 = self.lin(x1)
        v2 = self.lin(x1)
        v3 = self.lin(x1)
        return self.drop(torch.cat([v1, v2, v3], 1))
# Inputs to the model
x1 = torch.randn(1024, 256)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn_0 = torch.nn.BatchNorm1d(10240)
    def forward(self, x1):
        v1 = self.bn_0(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 10240)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(3, 3, 3, groups=3, bias=False)
        self.bn = torch.nn.BatchNorm1d(3)
    def forward(self, x2):
        s2 = self.conv(x2)
        s3 = self.bn(s2 + s2)
        return s3
# Inputs to the model
x2 = torch.randn(1, 3, 3)

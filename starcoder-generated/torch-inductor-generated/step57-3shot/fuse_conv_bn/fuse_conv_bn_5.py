
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(32, 19, 3, bias=False)
        self.conv_bn = torch.nn.BatchNorm1d(19)
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.conv_bn(x1)
        return x2 + x1
# Inputs to the model
x = torch.randn(1, 32, 19)

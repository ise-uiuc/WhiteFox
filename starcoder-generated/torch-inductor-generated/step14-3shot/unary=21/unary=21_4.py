
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(1000, 64, 13, stride=1, dilation=1, groups=1, bias=True)
        self.bn = torch.nn.BatchNorm1d(64)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 1000, 4802)

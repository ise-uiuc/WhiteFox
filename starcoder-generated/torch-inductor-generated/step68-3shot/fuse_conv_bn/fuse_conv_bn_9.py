
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv1d(64, 64, 5, stride=1, dilation=2, padding=2, groups=16, bias=True)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return x
# Inputs to the model
inputs = torch.randn(1, 64, 1000)

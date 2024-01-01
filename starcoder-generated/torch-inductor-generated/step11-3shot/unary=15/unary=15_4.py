
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        _conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=2, bias=False)
        self.conv = torch.nn.Sequential(
            _conv,
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            _conv,
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
        )
    def forward(self, x1):
        v1 = self.conv(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 32, 64)

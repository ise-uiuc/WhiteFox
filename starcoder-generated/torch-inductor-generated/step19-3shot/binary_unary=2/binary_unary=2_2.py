
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v1 = _F.hardsigmoid(v1, inplace=False)
        v2 = self.conv2(v1)
        v2 = _F.hardtanh(v2, min_val=0.1, max_val=0.7, inplace=False)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

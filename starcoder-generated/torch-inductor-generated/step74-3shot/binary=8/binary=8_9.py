
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, x2=None, x3=None):
        v1 = self.conv1(x1)
        if x2 is None:
            return torch.abs(v1)
        if x2 is not None and x3 is None:
            return v1 - torch.abs(v1)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 64, 64)

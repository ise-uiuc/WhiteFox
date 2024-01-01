
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 1, 1, stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.conv1(v1)
        return v2 - -9.582
# Inputs to the model
x = torch.randn(1, 2, 64, 64)

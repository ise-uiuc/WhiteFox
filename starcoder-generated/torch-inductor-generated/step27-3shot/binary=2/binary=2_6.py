
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, 1, stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(2, 2, 1, stride=1, padding=0)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.conv1(v1)
        return v2 - -8.582
# Inputs to the model
x = torch.randn(1, 3, 64, 64)

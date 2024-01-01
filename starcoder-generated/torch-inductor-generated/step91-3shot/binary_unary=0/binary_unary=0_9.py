
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=2, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, stride=2, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        return v1 + v2
# Inputs to the model
x = torch.randn(1, 16, 64, 64)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(16, 2, 7, stride=1, padding=3)
        self.conv1 = torch.nn.Conv2d(16, 2, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 16, 64, 64)

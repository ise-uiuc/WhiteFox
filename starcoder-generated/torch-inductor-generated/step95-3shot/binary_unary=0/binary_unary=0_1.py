
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 5, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
    def forward(self, x):
        self.conv1.groups = 32
        v1 = self.conv1(x)
        self.conv2.groups = 32
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 3, 256, 256)

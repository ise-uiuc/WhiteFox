
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 3, 3, stride=1, padding=1)
        self.maxpool = torch.nn.MaxPool2d(3)
        self.conv2 = torch.nn.Conv2d(3, 1, 1, stride=1, padding=0)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.maxpool(v1)
        v3 = self.conv2(v2)
        v4 = v3 > 1
        v5 = v3 * -0.9
        v6 = torch.where(v4, v3, v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 2, 32, 32)

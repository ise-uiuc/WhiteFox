
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(8, 64, 3, stride=1, padding=3)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = v1 + v1
        v3 = self.conv2(v2)
        v4 = v3 + x
        return v4
# Inputs to the model
x = torch.randn(1, 3, 224, 224)

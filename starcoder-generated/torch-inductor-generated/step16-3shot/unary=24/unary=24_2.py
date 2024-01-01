
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 7, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(8, 8, 3, stride=2, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = v3 > 0
        v5 = v3 * 0.1
        v6 = torch.where(v4, v3, v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)

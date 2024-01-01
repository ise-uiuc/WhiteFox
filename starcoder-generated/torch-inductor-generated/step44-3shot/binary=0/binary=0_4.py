
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(14, 16, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 6, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(16, 5, 3, stride=2, padding=1)
    def forward(self, x1, other=1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(x1)
        v4 = self.conv4(v3)
        if other == 1:
            other = torch.randn(v2.shape)
        v5 = v2 + v4 + other
        v6 = v2 + other
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

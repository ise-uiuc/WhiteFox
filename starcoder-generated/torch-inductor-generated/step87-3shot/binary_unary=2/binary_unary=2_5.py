
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.pool = torch.nn.AvgPool2d(3, stride=2)
        self.conv2 = torch.nn.Conv2d(3, 4, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.pool(v1)
        v3 = F.softmax(v2, dim=1)
        v4 = v3 - v3
        v5 = self.conv2(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

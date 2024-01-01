
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 5, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 7, 5, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v3 = F.max_pool2d(v1, 1)
        v2 = v3 - 0.5
        v4 = self.conv2(v2)
        v5 = F.avg_pool2d(v4, 3)
        v6 = F.softmax(v5, dim=1)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

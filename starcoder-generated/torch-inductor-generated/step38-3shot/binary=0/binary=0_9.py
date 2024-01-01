
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 5, 3, stride=1, padding=6)
    def forward(self, x1, x2, other1=1, other2=1):
        v1 = self.conv1(x1)
        v2 = torch.max_pool2d(v1, 3, stride=1, padding=5)
        v3 = self.conv2(v2)
        v4 = torch.mean(v3, dim=1, keepdim=True)
        return v4
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, stride=3, padding=4)
        self.conv2 = torch.nn.Conv2d(3, 5, 1, stride=1, padding=0)
    def forward(self, x1, x2, other1=1, other2=1):
        v1 = x1 + other2
        v2 = self.conv1(v1)
        v3 = v2 + other1
        v4 = torch.max_pool2d(v3, 3, stride=1, padding=5)
        v5 = self.conv2(v4)
        v6 = torch.mean(v5, dim=1, keepdim=True)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
x2 = torch.randn(1, 3, 10, 10)

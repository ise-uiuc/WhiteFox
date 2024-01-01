
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.conv3 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = self.relu1(v2)
        v4 = self.bn1(v1 + v3)
        v5 = self.bn2(v4)
        v6 = self.conv3(x1)
        s1 = v1.unsqueeze(0) * v6.unsqueeze(0).transpose(0, 2)
        (n, k) = s1.size()[-2:]
        s2 = s1.reshape(n, k, -1).sum(-1).div(k)
        return (v5, v6)
# Inputs to the model
x1 = torch.randn(1, 3, 32)
x2 = torch.randn(1, 3, 32)

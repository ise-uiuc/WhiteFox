
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(1, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv4(x1.squeeze())
        v2 = self.conv3(x1.squeeze())
        v3 = self.conv2(x1.squeeze())
        v4 = self.conv1(x1.squeeze())
        v5 = v1 + v2 + v3 + v4
        v6 = torch.relu(v5)
        return v6
x1 = torch.rand(1, 3, 64, 64)

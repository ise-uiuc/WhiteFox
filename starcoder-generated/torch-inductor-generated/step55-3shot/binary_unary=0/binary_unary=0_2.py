
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 32, 7, stride=2, padding=3)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 32, 7, stride=2, padding=3)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(32, 16, 7, stride=2, padding=3)
        self.bn3 = torch.nn.BatchNorm2d(16)
    def forward(self, x1, x2, x3):
        q1 = self.conv1(x1)
        h1 = self.bn1(q1)
        q2 = self.conv2(h1)
        h2 = self.bn2(q2)
        q3 = self.conv3(h2)
        h3 = self.bn3(q3)
        h4 = q2 + x2
        h5 = torch.relu(h4)
        h6 = h5 + q3
        h7 = torch.relu(h6)
        h8 = h7 + h1
        h9 = torch.relu(h8)
        h10 = h9 + q2
        h11 = torch.relu(h10)
        return h11
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3, x4):
        a1 = self.conv1(x1)
        v1 = a1 + x1
        a2 = self.conv2(v1)
        v2 = a2 + x2
        a3 = self.conv3(v2) + v2
        a4 = self.conv4(x3) + x2
        g1 = torch.relu(a3)
        g2 = a3 + a4
        g3 = torch.relu(a3)
        g4 = torch.relu(a4)
        g5 = self.conv3(a3)
        g6 = a4 + g5
        g7 = torch.relu(g6)
        return g7
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)

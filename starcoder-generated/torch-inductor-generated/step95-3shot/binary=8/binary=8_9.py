
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1, stride=1)
        self.relu = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, padding=1, stride=1)
        self.conv3 = torch.nn.Conv2d(32, 32, 3, padding=1, stride=1)
        self.conv4 = torch.nn.Conv2d(32, 32, 3, padding=1, stride=1)
        self.conv5 = torch.nn.Conv2d(32, 64, 3, padding=1, stride=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv6 = torch.nn.Conv2d(64, 64, 3, padding=1, stride=1)
        self.conv7 = torch.nn.Conv2d(64, 64, 1, padding=1, stride=1)
        self.bn3 = torch.nn.BatchNorm2d(64)
    def forward(self, x1, x2, x31, x32, x41, x42):
        y1 = self.conv1(x1)
        y2 = self.conv1(x2)
        w0 = self.relu(y1 + y2)
        w1 = self.bn1(w0)
        w2 = self.conv2(w1)
        w3 = self.conv3(w1) + self.conv4(w2)
        w4 = self.conv5(w3)
        w5 = self.relu(w4)
        y3 = self.bn2(w4)
        w6 = self.conv6(y3)
        w7 = self.conv7(w5)
        w8 = self.bn3(w6 + w7)
        return w8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 96, 96)
x4 = torch.randn(1, 3, 96, 96)

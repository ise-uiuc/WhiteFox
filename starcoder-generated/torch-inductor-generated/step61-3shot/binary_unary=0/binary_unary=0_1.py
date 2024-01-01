
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(64, 64, 13, stride=1, padding=6)
        self.conv3 = torch.nn.Conv2d(64, 64, 23, stride=1, padding=11)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = v1 + x1
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 + x2
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        v8 = v7 + x2
        v9 = torch.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 64, 128, 128)
x2 = torch.randn(1, 64, 128, 128)
x3 = torch.randn(1, 64, 128, 128)

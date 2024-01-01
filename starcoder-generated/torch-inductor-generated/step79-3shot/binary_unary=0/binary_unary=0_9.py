
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.conv5 = torch.nn.ConvTranspose2d(32, 32, 3, stride=2, padding=0)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 + x2
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        v8 = v7 + v2
        v9 = torch.relu(v8)
        v10 = self.conv4(v9)
        v11 = self.conv5(v10)
        v12 = v11 + v3
        v13 = torch.relu(v12)
        return v13
# Inputs to the model
x1 = torch.randn(1, 32, 640, 480)
x2 = torch.randn(1, 1,  240, 152)

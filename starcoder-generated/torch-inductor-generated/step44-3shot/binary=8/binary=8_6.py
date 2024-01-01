
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.fc = torch.nn.Linear(8*32, 16)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = v1 + v2
        v4 = self.conv3(x1)
        v5 = self.conv4(x2)
        v6 = self.conv5(x1) + v4 + v5
        v7 = self.conv5(x1)
        v8 = self.conv5(x2)
        v9 = v8 * v6
        v10 = v7 + v9
        v11 = v10.permute(0, 2, 1, 3)
        v12 = self.fc(v11)
        v13 = v12.permute(0, 1, 3, 2)
        return v13
# Inputs to the model
x1 = torch.randn(4, 3, 32, 32)
x2 = torch.randn(4, 3, 32, 32)

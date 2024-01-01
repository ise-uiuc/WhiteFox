
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(11, 13, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(13, 17, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(17, 19, 2, stride=1)
        self.conv4 = torch.nn.Conv2d(19, 23, 5, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 + x1
        v3 = self.conv2(v1)
        v4 = v2 + v3
        v5 = torch.relu(v3)
        v6 = self.conv3(v4)
        v7 = v5 + v2
        v8 = torch.relu(v3)
        v9 = self.conv4(v6)
        v10 = v8 + v2
        v11 = v7 + v10
        return v11
# Input to the model.
x1 = torch.randn(1, 11, 229, 224)

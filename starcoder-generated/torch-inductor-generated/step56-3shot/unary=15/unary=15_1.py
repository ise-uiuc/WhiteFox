
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 21, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(21, 31, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(31, 41, 3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(41, 51, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(51, 61, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v1)
        v4 = torch.relu(v3)
        v5 = self.conv3(v3)
        v6 = torch.relu(v5)
        v7 = self.conv4(v5)
        v8 = torch.relu(v7)
        v9 = self.conv5(v7)
        v10 = torch.relu(v9)
        return v10
# Inputs to the model
x1 = torch.randn(4, 1, 1760, 1760)

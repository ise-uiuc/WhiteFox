
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 8, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(32, 4, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(4, 10, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.max_pool2d(v3, kernel_size=3, stride=3)
        v5 = self.conv3(v4)
        v6 = torch.relu(v5)
        v7 = torch.cat([v6, v6], axis=-1)
        v8 = self.conv4(v7)
        v9 = torch.relu(v8)
        v10 = self.conv5(v9)
        v11 = torch.relu(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)

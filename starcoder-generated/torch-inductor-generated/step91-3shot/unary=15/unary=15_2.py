
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(6, 16, 5, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(16, 120, 5, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(120, 84, 5, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = torch.max_pool2d(v2, kernel_size=2, stride=2, padding=0)
        v4 = self.conv2(v3)
        v5 = torch.relu(v4)
        v6 = torch.max_pool2d(v5, kernel_size=2, stride=2, padding=0)
        v7 = self.conv3(v6)
        v8 = torch.sigmoid(v7)
        v9 = self.conv4(v8)
        v10 = torch.sigmoid(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)

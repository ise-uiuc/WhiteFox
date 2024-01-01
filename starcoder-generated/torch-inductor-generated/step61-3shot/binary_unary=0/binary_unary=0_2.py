
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(3, 3, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(3, 3, 7, stride=1, padding=3)
        def forward(self, x1, x2, x3):
            v1 = self.conv1(x1)
            v2 = self.conv3(x2)
            v3 = v1 + v2
            v4 = torch.relu(v3)
            v5 = self.conv2(v4)
            v6 = v5 + x3
            v7 = torch.relu(v6)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 64, 64)

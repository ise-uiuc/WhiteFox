
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 5, stride=2, padding=2)

        self.conv2 = torch.nn.Conv2d(16, 48, 5, stride=2, padding=2)

        self.conv3 = torch.nn.Conv2d(48, 96, 5, stride=1, padding=1, groups=2)

        self.conv4 = torch.nn.Conv2d(96, 192, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = F.relu(self.conv1(x))
        v2 = F.relu(self.conv2(v1))
        v3 = F.relu(self.conv3(v2))
        v4 = F.relu(self.conv4(v3))

        v5 = F.max_pool3d(v4, kernel_size=3, stride=3, padding=1)
        return v5
# Inputs to the model
x = torch.randn(3, 1, 10, 30, 20)

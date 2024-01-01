
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = torch.nn.Conv2d(32, 32, (2, 2), (3, 3), (2, 2))
        self.conv2d_2 = torch.nn.Conv2d(32, 24, (4, 2), (1, 7), (0, 0))
        self.conv2d_3 = torch.nn.Conv2d(24, 8, (1, 7), (1, 1), (0, 3))
    def forward(self, x1):
        v1 = self.conv2d_1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2d_2(v2)
        v4 = torch.relu(v3)
        v5 = self.conv2d_3(v4)
        v6 = torch.relu(v5)
        v7 = torch.softmax(v6, dim=1)
        return v7
# Inputs to the model
x1 = torch.randn(1, 32, 35, 20)

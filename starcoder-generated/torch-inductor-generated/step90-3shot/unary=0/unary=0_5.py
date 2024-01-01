
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(9, 10, 3, stride=18, padding=3)
        self.conv3 = torch.nn.Conv2d(9, 10, 5, stride=2, padding=0)
    def forward(self, x9):
        v1 = self.conv1(x9)
        v2 = self.conv2(v1)
        v3 = self.conv3(v1)
        v4 = v2 * 0.5
        v5 = v3 * v2
        v6 = v2 * v3
        v7 = v3 * v4
        v8 = v6 * v5
        v9 = v7 * v4
        v10 = v5 * v7
        return v10
# Inputs to the model
x9 = torch.randn(1, 3, 111, 97)

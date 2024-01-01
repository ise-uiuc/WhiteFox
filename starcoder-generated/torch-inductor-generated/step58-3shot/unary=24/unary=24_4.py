
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 9, 3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(13, 9, 3, stride=2, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(13, 3, 3, stride=1, padding=1)
    def forward(self, x):
        negative_slope = 0.12843823
        v1 = self.conv1(x)
        v2 = self.relu1(v1)
        v3 = self.conv2(v2)
        v4 = self.relu2(v3)
        v5 = self.conv3(v4)
        v6 = v5 > 0
        v7 = v5 * negative_slope
        v8 = torch.where(v6, v5, v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 2, 108, 55)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(17, 10, 2, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(10, 20, 2, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(20, 70, 2, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(70, 60, 2, stride=2, padding=1)
    def forward(self, x1, x2, other=None):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        if other == None:
            other = v4
        v5 = F.relu(v4 + other)
        return v5
# Inputs to the model
x1 = torch.randn(1, 17, 64, 64)
x2 = torch.randn(1, 17, 64, 64)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        # self.conv2 = torch.nn.Conv2d(3, 8, 3,stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv3(torch.add(x, v1))
        v3 = self.conv2(torch.add(x, v1)) # Missing a layer
        v4 = self.conv4(torch.add(x, v3))
        v5 = v3 + v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)

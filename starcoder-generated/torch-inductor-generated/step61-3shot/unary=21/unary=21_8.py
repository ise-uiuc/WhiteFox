
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 3)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(16, 32, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(32, 32, 3, 1, 1)
        self.conv5 = torch.nn.Conv2d(32, 32, 3, 1, 1)
    def forward(self, x):
        v1 = torch.tanh(self.conv1(x))
        v2 = self.conv2(x)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        return v5
# Inputs to the model
x = torch.randn(1, 1, 28, 28)

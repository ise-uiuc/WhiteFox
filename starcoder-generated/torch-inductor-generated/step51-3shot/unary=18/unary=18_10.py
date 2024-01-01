
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 5, 1, 2)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, 2, 1)
        self.conv3 = torch.nn.Conv2d(8, 8, 3, 1, 0)
        self.conv4 = torch.nn.Conv2d(8, 1, 3, 1, 2)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 1, 64, 64)

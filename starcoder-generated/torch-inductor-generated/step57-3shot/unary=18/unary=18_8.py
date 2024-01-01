
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 4, 5, 1, 1)
        self.conv2 = torch.nn.Conv2d(4, 4, 5, 1, 1)
        self.conv3 = torch.nn.Conv2d(4, 4, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(4, 4, 3, 1, 1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v1 = self.conv2(v1)
        v1 = self.conv3(v1)
        v1 = self.conv4(v1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 8, 32, 64)

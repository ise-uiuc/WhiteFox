
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(4, 4, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(4, 4, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(4, 4, 3, 1, 1)
    def forward(self, x1):
        x2 = torch.sigmoid(self.conv1(x1))
        x3 = torch.sigmoid(self.conv2(x2))
        x4 = torch.sigmoid(self.conv3(x3))
        x5 = torch.sigmoid(self.conv4(x4))
        return x5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

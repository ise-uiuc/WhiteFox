
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 128, 5, 1, 2)
        self.conv2 = torch.nn.Conv2d(128, 64, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(64, 4, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(4, 1, 3, 1, 1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v1 = torch.sigmoid(v1)
        v2 = self.conv2(v1)
        v2 = torch.sigmoid(v2)
        v3 = self.conv3(v2)
        v3 = torch.sigmoid(v3)
        v4 = self.conv4(v3)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 128, 128)

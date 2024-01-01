
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(10, 10, 5, 1, 2)
        self.conv3 = torch.nn.Conv2d(10, 10, 1, 2, 2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.sigmoid(v2)
        v4 = self.conv3(v3)
        v5 = torch.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

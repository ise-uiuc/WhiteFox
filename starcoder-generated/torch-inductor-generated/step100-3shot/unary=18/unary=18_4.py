
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 9, (13, 13), 1, (0, 1))
        self.conv2 = torch.nn.Conv2d(9, 3, (13, 13), 1, (1, 5))
        self.conv3 = torch.nn.Conv2d(3, 3, (13, 13), 1, (0, 2))
    def forward(self, x2):
        v1 = torch.sigmoid(self.conv1(x2))
        v3 = torch.sigmoid(self.conv2(v1))
        v5 = torch.sigmoid(self.conv3(x2))
        return None
# Inputs to the model
x2 = torch.randn(1, 3, 9, 28)

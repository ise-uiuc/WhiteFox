
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, (3, 3), 1, (1, 1))
        self.conv2 = torch.nn.Conv2d(3, 3, (1, 3), 1, padding=(1, 2))
        self.conv3 = torch.nn.Conv2d(3, 3, (4, 3), 1, padding=(2, 1))
    def forward(self, x1):
        v1 = torch.sigmoid(self.conv1(x1))
        v2 = torch.sigmoid(self.conv2(v1))
        v3 = torch.sigmoid(self.conv3(v2))
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 9, 28)

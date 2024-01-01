
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, 11, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(10, 10, 7, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(10, 10, 5, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(10, 10, 3, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1) - 10
        v3 = torch.relu(v2)
        v4 = self.conv3(v3) - 10
        v5 = torch.tanh(v4)
        v6 = self.conv4(v5) - 10
        v7 = torch.sigmoid(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

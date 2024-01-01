
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 11, 3, stride=1, padding=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.conv2 = torch.nn.Conv2d(11, 32, 1, stride=1, padding=0)
        self.sigmoid_1 = torch.nn.Sigmoid()
        self.conv3 = torch.nn.Conv2d(32, 128, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = self.sigmoid_1(v3)
        v5 = self.conv3(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 6, 9, 9)

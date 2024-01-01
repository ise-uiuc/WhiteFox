
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 5, 1, stride=1, padding=0)
        self.pool = torch.nn.MaxPool2d((3, 3), stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(5, 28, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(5, 28, 3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(13, 28, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.pool(v1)
        v3 = self.conv2(v2)
        v4 = self.conv3(v2)
        v5 = torch.cat((v3, v4), 1)
        v6 = self.conv4(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 2, 24, 24)

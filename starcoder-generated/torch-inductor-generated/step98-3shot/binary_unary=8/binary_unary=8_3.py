
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, (1, 3), stride=1, padding=(0, 1))
        self.conv2 = torch.nn.Conv2d(3, 8, (1, 5), stride=1, padding=(0, 2))
        self.conv3 = torch.nn.Conv2d(3, 8, (1, 7), stride=1, padding=(0, 3))
        self.conv4 = torch.nn.Conv2d(3, 8, (3, 1), stride=1, padding=(1, 0))
        self.conv5 = torch.nn.Conv2d(3, 8, (5, 1), stride=1, padding=(2, 0))
        self.conv6 = torch.nn.Conv2d(3, 8, (7, 1), stride=1, padding=(3, 0))
        self.conv7 = torch.nn.Conv2d(3, 8, (1, 1), stride=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = self.conv3(x)
        v4 = self.conv4(x)
        v5 = self.conv5(x)
        v6 = self.conv6(x)
        v7 = self.conv7(x)
        v8 = v1 + v2 + v3 + v4 + v5 + v6 + v7
        return v8
# Inputs to the model
x = torch.randn(1, 3, 64, 64)

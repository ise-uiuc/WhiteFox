
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, (3, 3), stride=1, padding=(1, 1))
        self.conv2 = torch.nn.Conv2d(8, 4, (3, 3), stride=1, padding=(1, 1))
        self.conv3 = torch.nn.Conv2d(4, 8, (3, 3), stride=1, padding=(1, 1))
        self.conv4 = torch.nn.Conv2d(8, 4, (3, 3), stride=1, padding=(1, 1))
        self.conv5 = torch.nn.Conv2d(4, 1, (1, 1), stride=1, padding=(0, 0))
    def forward(self, x1, other):
        x1 = self.conv1(x1)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = x5 + other
        return x6
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)

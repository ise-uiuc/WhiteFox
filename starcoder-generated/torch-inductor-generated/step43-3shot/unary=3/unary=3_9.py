    Note that the outputs of conv5 and conv6 are different at the position (0, 0), so if the error function is used, this position will obtain the largest.
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 10, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(10, 12, 3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(12, 4, 3, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(4, 1, 3, stride=2, padding=1)
        self.conv6 = torch.nn.Conv2d(1, 2, 3, stride=2, padding=1)
        self.conv7 = torch.nn.Conv2d(2, 1, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        v6 = self.conv6(v4)
        v7 = self.conv7(v6)
        return v1, v5, v7
# Inputs to the model
x1 = torch.randn(1, 8, 84, 84)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(7, 16, 1, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 64, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
    def forward(self, data, other1, other2=1, testcase="0"):
        v1 = self.conv1(data)
        v2 = other1.permute(0, 3, 1, 2)
        if testcase == "0":   # A test case that generates conv fusion
            v3 = self.conv2(v2)
            v4 = other2.permute(0, 3, 1, 2)
            x3 = self.conv3(v3)
            x4 = self.conv4(x3)
            x5 = self.conv5(x4)
            x6 = self.conv6(x5)
            v5 = self.conv2(v1)
            output = v5 + x6
        else:
            output = self.conv2(v1)
        return output
# Inputs to the model
data = torch.randn(2, 7, 64, 64)
other1 = torch.randn(2, 2, 16, 16)
other2 = torch.randn(2, 1, 16, 16)

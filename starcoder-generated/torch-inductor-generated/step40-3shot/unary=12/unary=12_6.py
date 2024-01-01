
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(6, 8, (3, 7), stride=(1, 3), padding=(1, 2), dilation=(1, 2), groups=(1, 2), bias=True)
        self.conv2 = torch.nn.Conv2d(8, 8, (3, 7), stride=(1, 3), padding=(1, 2), dilation=(1, 2), groups=(1, 2), bias=True)
        self.sigmoid1 = torch.nn.Sigmoid()
        self.sigmoid2 = torch.nn.Sigmoid()
        self.conv3 = torch.nn.Conv2d(8, 12, (3, 4), stride=(1, 3), padding=(1, 3), dilation=(1, 2))
        self.conv4 = torch.nn.Conv2d(12, 16, 1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.sigmoid1(v1)
        v3 = v1 * v2
        v4 = self.conv2(v3)
        v5 = self.sigmoid2(v4)
        v6 = v4 * v5
        v7 = v6 - v1
        v8 = self.conv3(v7)
        v9 = self.conv4(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 32, (1, 1), stride=(2,2), bias=False)
        self.conv1 = torch.nn.Conv2d(32, 32, (1, 1), stride=(1,1), bias=False)
        self.conv2 = torch.nn.Conv2d(32, 32, (1, 1), stride=(1,1), bias=False)
        self.conv3 = torch.nn.Conv2d(32, 32, (1, 1), stride=(1,1), bias=False)
        self.conv4 = torch.nn.Conv2d(32, 32, (1, 1), stride=(1,1), bias=False)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.sigmoid(v1)
        v3 = self.conv1(v2)
        v4 = self.sigmoid(v3)
        v5 = self.conv2(v4)
        v6 = self.sigmoid(v5)
        v7 = self.conv3(v6)
        v8 = self.sigmoid(v7)
        v9 = self.conv4(v8)
        v10 = v9 * v2
        return v10
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)

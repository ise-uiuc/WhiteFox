
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, (2, 2), stride=(1, 1), padding=(1, 0))
        self.conv2 = torch.nn.Conv2d(32, 32, (1, 2), stride=(1, 1), padding=(0, 1))
        self.conv3 = torch.nn.Conv2d(32, 32, (1, 2), stride=(1, 1), padding=(1, 0))
        self.conv4 = torch.nn.Conv2d(32, 32, (1, 2), stride=(1, 1), padding=(0, 1))
        self.conv5 = torch.nn.Conv2d(32, 1, (1, 1), stride=(1, 1), padding=(0, 0))
    def forward(self, x4):
        v1 = self.conv1(x4)
        v2 = self.conv2(v1)
        v3 = self.conv3(v1)
        v4 = self.conv4(v2)
        v5 = self.conv5(v3)
        v6 = torch.sigmoid(v5)
        return v6
# Inputs to the model
x4 = torch.randn(6, 3, 64, 64)

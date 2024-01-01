
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(18, 15, (3, 9), stride=(1, 5), padding=(1, 0))
        self.conv2 = torch.nn.Conv2d(15, 12, (3, 8), stride=(1, 3), padding=(1, 0))
        self.conv3 = torch.nn.Conv2d(12, 9, (5, 3), stride=(1, 3), padding=(0, 0))
        self.conv4 = torch.nn.Conv2d(9, 6, (5, 3), stride=(1, 5), padding=(0, 2))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv3(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.conv4(v6)
        v8 = torch.sigmoid(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 18, 48, 48)
# Model begins
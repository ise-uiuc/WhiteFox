
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, (7, 1), stride=(1, 1))
        self.conv2 = torch.nn.Conv2d(6, 22, (2, 3), stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(22, 34, (2, 3), stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(34, 29, (2, 3), stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(29, 39, (2, 2), stride=(2, 2))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.sigmoid(v2)
        v4 = self.conv3(v3)
        v5 = self.conv4(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.conv5(v6)
        v8 = torch.sigmoid(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 1, 7, 7)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(103, 204, (16, 6), stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(204, 897, (16, 1), stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(897, 654, (16, 15), stride=1, padding=7)
        self.conv4 = torch.nn.Conv2d(654, 768, (16, 7), stride=23, padding=3)
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
x1 = torch.randn(1, 103, 67, 159)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 40, (1, 1), stride=(2, 2), padding=0, bias=False)
        self.conv2 = torch.nn.Conv2d(40, 80, (3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(80, 1, (1, 1), stride=(1, 1), padding=0, bias=False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = torch.sigmoid(v1 + v2)
        v4 = self.conv2(v3)
        v5 = self.conv2(v3)
        v6 = self.conv2(v3)
        v7 = v4 + v5 + v6
        v8 = self.conv3(v7)
        v9 = self.conv3(v7)
        v10 = self.conv3(v7)
        v11 = v8 + v9 + v10
        return v11
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)

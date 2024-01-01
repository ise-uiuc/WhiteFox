
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(8, 1, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        v6 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        v7 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        v8 = torch.nn.Conv2d(8, 1, 1, stride=1, padding=1)
        v9 = self.conv4(v8)
        v10 = torch.sigmoid(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)

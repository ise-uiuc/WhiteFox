
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 4, 1, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(4, 8, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(4, 8, 1, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(8, 8, 1, stride=2, padding=1)
        self.conv7 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.conv8 = torch.nn.Conv2d(8, 8, 1, stride=2, padding=1)
        self.conv9 = torch.nn.Conv2d(8, 8, 5, stride=1, padding=2)
        self.conv10 = torch.nn.Conv2d(8, 8, 1, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(v1)
        v4 = self.conv4(v2)
        v5 = self.conv5(v3)
        v6 = self.conv6(v4)
        v7 = self.conv7(v5)
        v8 = self.conv8(v6)
        v9 = self.conv9(v7)
        v10 = self.conv10(v8)
        return v9
# Inputs to the model.
x1 = torch.randn(1, 3, 64, 64)

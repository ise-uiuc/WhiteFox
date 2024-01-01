
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(100, 100, (1, 1), stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(100, 100, (3, 1), stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(100, 80, (5, 1), stride=1, padding=2)
        self.conv4 = torch.nn.Conv2d(80, 80, (1, 3), stride=1, padding=(0, 1))
        self.conv5 = torch.nn.Conv2d(80, 80, (1, 1), stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(80, 80, (3, 1), stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(80, 80, (5, 1), stride=1, padding=2)
        self.conv8 = torch.nn.Conv2d(80, 80, (1, 3), stride=1, padding=(0, 1))
        self.conv9 = torch.nn.Conv2d(80, 184, (1, 1), stride=1, padding=0)
        self.conv10 = torch.nn.Conv2d(184, 72, (1, 1), stride=1, padding=0)
        self.conv11 = torch.nn.Conv2d(72, 12, (1, 1), stride=1, padding=0)
        self.conv12 = torch.nn.Conv2d(12, 5, (1, 1), stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1) * 0.5
        v2 = self.conv2(v1) + 1
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        v6 = self.conv6(v5)
        v7 = self.conv7(v6)
        v8 = self.conv8(v7)
        v9 = self.conv9(v8)
        v10 = self.conv10(v9)
        v11 = self.conv11(v10)
        v12 = self.conv12(v11)
        return v12
# Inputs to the model
x1 = torch.randn(1, 100, 50, 50)

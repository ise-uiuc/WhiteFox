
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 4, 5, stride=2, padding=2)
        self.conv3 = torch.nn.Conv2d(4, 32, 2, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(32, 8, 3, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(16, 64, 1, stride=1, padding=0)
        self.conv7 = torch.nn.Conv2d(64, 64, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.nn.functional.gelu(v1)
        v3 = self.conv2(v2)
        v4 = torch.nn.functional.gelu(v3)
        v5 = self.conv3(v4)
        v6 = torch.nn.functional.gelu(v5)
        v7 = self.conv4(v6)
        v8 = torch.nn.functional.gelu(v7)
        v9 = self.conv5(v8)
        v10 = torch.nn.functional.gelu(v9)
        v11 = self.conv6(v10)
        v12 = torch.nn.functional.gelu(v11)
        v13 = self.conv7(v12)
        return v13
# Inputs to the model
x1 = torch.randn(1, 32, 18, 18)

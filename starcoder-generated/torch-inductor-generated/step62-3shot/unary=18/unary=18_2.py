
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv11 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv12 = torch.nn.Conv2d(32, 32, kernel_size=1, padding=0)
        self.conv13 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv21 = torch.nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv22 = torch.nn.Conv2d(32, 32, kernel_size=1, padding=0)
        self.conv31 = torch.nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv32 = torch.nn.Conv2d(32, 3, kernel_size=1, padding=0)
    def forward(self, x1):
        v1 = torch.relu(self.conv11(x1))
        v2 = self.conv12(v1)
        v3 = self.conv13(v2)
        v4 = torch.relu(self.conv21(v3))
        v5 = self.conv22(v4)
        v6 = torch.relu(self.conv31(v5))
        v7 = self.conv32(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)

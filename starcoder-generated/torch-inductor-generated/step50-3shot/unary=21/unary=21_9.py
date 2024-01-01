
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 24, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(24, 100, 1, stride=1)
        self.conv3 = torch.nn.Conv2d(100, 64, 1, stride=1)
        self.conv4 = torch.nn.Conv2d(64, 1, 3, stride=1,
            padding=1)
        self.hardtanh_1 = torch.nn.Hardtanh()
        self.hardtanh_2 = torch.nn.Hardtanh()
        self.tanh = torch.nn.Tanh()
        self.flatten = torch.nn.Flatten(1, -1)
        self.linear1 = torch.nn.Linear(200, 300)
        self.linear2 = torch.nn.Linear(300, 1)
    def forward(self, x97):
        x96 = self.conv1(x97)
        x19 = self.conv2(x96)
        x3 = self.conv3(x19)
        x9 = self.conv4(x3)
        x10 = self.hardtanh_1(x9)
        x11 = self.hardtanh_2(x10)
        x12 = self.hardtanh_2(x11)
        x13 = self.hardtanh_1(x12)
        x14 = self.hardtanh_1(x3)
        x15 = self.hardtanh_1(x14)
        x16 = self.hardtanh_2(x15)
        x5 = self.tanh(x16)
        x32 = self.flatten(x5)
        x33 = self.linear1(x32)
        x95 = self.tanh(x33)
        x98 = self.tanh(x95)
        x119 = self.linear2(x98)
        return x119
# Inputs to the model
x97=torch.randn(4, 3, 224, 224)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = torch.nn.Conv2d(3, 64, 3, stride=2, padding=1)
        self.conv1_2 = torch.nn.Conv2d(3, 64, 1, stride=1, padding=0)
        self.maxpool1 = torch.nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2_1 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.maxpool2 = torch.nn.MaxPool2d(3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.maxpool1(self.conv1_1(x1))
        v2 = self.conv1_2(x1)
        v3 = v2 + v1
        v4 = torch.clamp(v3, 0, 6)
        v5 = self.maxpool2(self.conv2_1(v4))
        v6 = self.conv2_2(v4)
        v7 = v6 + v5
        v8 = torch.clamp(v7, 0, 6)
        return v8.permute(0, 3, 1, 2)
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)

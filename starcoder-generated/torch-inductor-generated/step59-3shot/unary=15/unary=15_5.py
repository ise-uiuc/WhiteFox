
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 16, kernel_size=7, padding=3, stride=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=7, padding=3, stride=1)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.conv4 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.tanh(v1)
        v3 = self.pool1(v2)
        v4 = self.conv2(v3)
        v5 = torch.tanh(v4)
        v6 = self.conv3(v5)
        v7 = torch.tanh(v6)
        v8 = self.conv4(v7)
        v9 = torch.tanh(v8)
        v10 = self.pool1(v9)
        return v10
# Inputs to the model
x1 = torch.randn(3, 32, 224, 224)

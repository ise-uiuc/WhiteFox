
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 32, 7, 3)
        self.conv2d_1 = torch.nn.Conv2d(32, 48, 5, 1)
        self.conv2d_2 = torch.nn.ConvTranspose2d(48, 64, 2, 1)
        self.maxpool2d = torch.nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
        self.avgpool2d = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=0)
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax()
    def forward(self, x):
        v1 = self.conv2d(x)
        v2 = self.conv2d_1(v1)
        v3 = self.conv2d_2(v2)
        v4 = self.maxpool2d(v3)
        v5 = self.avgpool2d(v4)
        x1 = self.tanh(v5)
        x2 = self.softmax(x1)
        return x2
# Inputs to the model
x = torch.randn(64, 3, 112, 112)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 64, kernel_size=(6, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=(100, 3), padding=18, dilation=4)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 32, 17, 17)

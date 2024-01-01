
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = torch.nn.ReLU6()
        self.conv2d = torch.nn.Conv2d(640, 1280, 1, 1)
        self.maxpool2d = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v3 = self.maxpool2d(v1)
        v2 = self.relu6(v3)
        return v2
# Inputs to the model
x1 = torch.randn(1, 640, 7, 7)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=15, stride=1, padding=7, dilation=2)
        self.sigm1 = torch.nn.Sigmoid()
        self.conv2 = torch.nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, dilation=1)
        self.sigm2 = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.sigm1(v1)
        v3 = self.conv2(v2)
        v4 = self.sigm2(v3)
        v5 = v1 * v3 # Multiply the output of the sigmoid function to the output of the convolution
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

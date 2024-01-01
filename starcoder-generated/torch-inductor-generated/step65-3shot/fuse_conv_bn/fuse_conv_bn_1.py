
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.pool = torch.nn.MaxPool2d(kernel_size=10, stride=3, padding=2, dilation=5)
        torch.manual_seed(1)
        self.conv = torch.nn.Conv2d(1, 1, 1)
    def forward(self, x1):
        v1 = self.pool(x1)
        return self.pool(self.conv(x1))
# Inputs to the model
x1 = torch.randn(1, 1, 10, 10)

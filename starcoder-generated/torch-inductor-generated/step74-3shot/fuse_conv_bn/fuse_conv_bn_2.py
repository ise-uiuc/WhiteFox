
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=3)
        self.pool = torch.nn.MaxPool2d(kernel_size=2)
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 32, 32)

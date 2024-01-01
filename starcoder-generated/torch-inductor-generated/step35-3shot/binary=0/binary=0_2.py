
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, 2, stride=2, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.pool(v1)
        return v2

# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)

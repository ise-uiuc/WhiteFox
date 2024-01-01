
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    def forward(self, x1):
        v1 = self.max1(x1)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 3, 3, 1, 1)
        self.maxpool2d = torch.nn.MaxPool2d(3, 2, 0, 1)
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = self.maxpool2d(v1)
        return (v2, self.conv2d(x1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.pool1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.pool2(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)

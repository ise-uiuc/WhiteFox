
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 3, stride=1, padding=1, bias=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.maxpool(v1)
        v3 = torch.add(v2, x2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 256, 64)
x2 = torch.randn(500, 6, 256, 64)

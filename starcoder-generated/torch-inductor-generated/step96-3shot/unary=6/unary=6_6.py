
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 5, stride=1, padding=2)
        self.max_pool2d = torch.nn.MaxPool2d(3, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = self.max_pool2d(t1)
        return t2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

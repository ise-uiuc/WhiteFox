
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.conv = torch.nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.avg = torch.nn.AvgPool2d(4, stride=4, padding=59)
    def forward(self, x):
        v = self.conv(x)
        v = self.sigmoid(v)
        v = self.avg(v)
        return v
# Inputs to the model
x = torch.randn(1, 3, 64, 64)

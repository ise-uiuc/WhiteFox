
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 10, 1, stride=1, padding=1, bias=True)
        self.pool = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    def forward(self, x1, padding2=None):
        v1 = self.conv(x1)
        v1 = self.pool(v1)
        if padding2 == None:
            padding2 = torch.randn(v1.shape)
        v2 = v1 + padding2
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

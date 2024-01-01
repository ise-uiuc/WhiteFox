
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=128, kernel_size=[30, 30], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False)
        self.conv = torch.nn.parameter.Parameter(torch.ones_like(self.conv))
    def forward(self, x1):
        x2 = torch.nn.functional.dropout(x1, p=0.3)
        x3 = torch.rand_like(x1)
        x4 = self.conv(x2)
        return x4
# Inputs to the model
x1 = torch.randn(1, 1, 28, 20)

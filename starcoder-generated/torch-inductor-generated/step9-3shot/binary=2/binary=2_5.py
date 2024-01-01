
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=1, groups=1, bias=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 1
        return v2
# Inputs to the model
x1 = torch.randn(1, 5, 12, 1)

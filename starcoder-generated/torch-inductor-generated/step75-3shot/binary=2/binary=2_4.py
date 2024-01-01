
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 5, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=1, groups=1, bias=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - x1
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)

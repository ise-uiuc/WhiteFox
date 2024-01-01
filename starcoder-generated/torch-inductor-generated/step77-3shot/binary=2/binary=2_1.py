
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 48, kernel_size=(1,5), stride=(1,5), padding=(20,0), dilation=1, groups=1, bias=False)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 1.0
        return v2
# Inputs to the model
x = torch.randn(1, 32, 64, 64)

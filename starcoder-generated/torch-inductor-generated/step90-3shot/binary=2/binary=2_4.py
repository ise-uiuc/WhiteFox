
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=5, stride=(2,2), padding=(4,4), dilation=(4,4), groups=4)
    def forward(self, x):
        t1 = self.conv(x)
        t2 = t1 - 1
        return t2
# Inputs to the model
x = torch.randn(1, 3, 32, 32)

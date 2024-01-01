
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    def forward(self, x1, kernel, other, group=1):
        v1 = self.conv(x1)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
kernel = torch.randn(3, 3)
other = 1

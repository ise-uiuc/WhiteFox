
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=11, kernel_size=(1, 2), stride=(3, 1), groups=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        return v1
# Inputs to the model
x1 = torch.randn(4, 3, 13, 10)

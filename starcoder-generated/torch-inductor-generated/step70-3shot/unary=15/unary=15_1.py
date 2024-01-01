
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=2)
    def forward(self, x2):
        v4 = torch.mul(torch.add(self.conv(x2), x2), x2)
        return v4
# Inputs to the model
x2 = torch.randn(6, 3, 128, 128)

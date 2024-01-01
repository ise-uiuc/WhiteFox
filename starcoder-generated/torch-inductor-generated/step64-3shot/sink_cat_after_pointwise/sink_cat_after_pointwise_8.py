
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.cat([torch.randn((1, 20, 10, 10)), torch.randn((1, 20, 10, 10)), torch.randn((1, 20, 10, 10))], dim=1)
        self.conv = torch.nn.Conv2d(in_channels=20, out_channels=64, kernel_size=4, stride=4)
    def forward(self, x):
        x = self.t1
        x = self.conv(x)
        return x.view(x.shape[0], -1)
# Inputs to the model
x = torch.randn(1)

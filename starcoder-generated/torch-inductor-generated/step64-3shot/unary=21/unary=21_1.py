
class CNNModel1(torch.nn.Module):
    def __init__(self):
        super(CNNModel1, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, groups=1)
    def forward(self, x):
        x = x.float()
        x = self.conv(x)
        return x
# Inputs to the model
x = torch.randn(20, 1, 12, 12)

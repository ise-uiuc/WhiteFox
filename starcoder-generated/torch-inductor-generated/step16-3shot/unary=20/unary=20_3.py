
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = torch.nn.Sequential(torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=2, padding=0), torch.nn.LeakyReLU(negative_slope = 0.2, inplace = False), torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(negative_slope = 0.2, inplace = False), torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=2, padding=0), torch.nn.LeakyReLU(negative_slope = 0.2, inplace = False), torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(negative_slope = 0.2, inplace = False))
    def forward(self, x1):
        v1 = self.convs(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)

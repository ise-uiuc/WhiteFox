
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(in_channels=256, out_channels=32, kernel_size=3, stride=(1, 1), padding=(1, 1)), torch.nn.ReLU())
    def forward(self, x1):
        v1 = torch.nn.functional.interpolate(x1, scale_factor=0.125)
        v2 = self.conv(v1)
        v3 = torch.nn.functional.interpolate(v2, scale_factor=8.0)
        return v3
# Inputs to the model
x1 = torch.randn(1, 256, 9, 11)

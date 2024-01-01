
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d1 = torch.nn.Conv2d(1, 1, 2)
        self.batch_norm1 = torch.nn.BatchNorm2d(1)
        self.conv_transpose2d1 = torch.nn.ConvTranspose2d(1, 1, 1)
        self.interpolate1 = torch.nn.Upsample(scale_factor=1)
    def forward(self, x):
        x = self.conv2d1(x)
        x = self.batch_norm1(x)
        x = self.conv_transpose2d1(x)
        y = self.interpolate1(x)
        return y
# Inputs to the model
x = torch.randn(1, 1, 4, 4)

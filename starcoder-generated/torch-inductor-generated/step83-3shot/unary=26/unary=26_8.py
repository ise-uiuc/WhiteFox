
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(8, 16, 1, stride=1, padding=0)
        self.conv_t2 = torch.nn.ConvTranspose2d(16, 8, 1, stride=1, padding=0)
        self.conv_t3 = torch.nn.ConvTranspose2d(8, 8, 1, stride=1, padding=0)
    def forward(self, x12):
        x13 = self.conv_t1(x12)
        x14 = self.conv_t2(x13)
        x15 = self.conv_t3(x14)
        x16 = x15 > 0
        x17 = x15 * 16
        x18 = torch.where(x16, x15, x17)
        return torch.nn.functional.interpolate(x18, scale_factor=[2.0, 2.0])
# Inputs to the model
x12 = torch.randn(31, 8, 384, 384)

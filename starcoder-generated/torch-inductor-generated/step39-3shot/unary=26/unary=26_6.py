
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(2, 6, 5, stride=2, padding=1, bias=False)
    def forward(self, x12):
        x1 = self.conv_t(x12)
        x2 = x1 > 0
        x3 = x1 * -9.821
        x4 = torch.where(x2, x1, x3)
        x5 = torch.transpose(x4, 1, 3)
        x6 = torch.transpose(x4, 2, 3)
        return torch.nn.functional.pixel_shuffle(x5, 3)
# Inputs to the model
x12 = torch.randn(2, 2, 24, 99)

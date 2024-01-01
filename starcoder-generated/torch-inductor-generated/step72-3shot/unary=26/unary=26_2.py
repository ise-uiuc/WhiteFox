
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(16, 7, 3, stride=3, padding=1, output_padding=0)
    def forward(self, x1):
        f1 = self.conv_t(x1)
        f2 = f1 > 0
        f3 = f1 * -0.511
        f4 = torch.where(f2, f1, f3)
        return torch.nn.functional.interpolate(torch.nn.functional.relu(f4), size=(1, 1), mode='linear', align_corners=False)
# Inputs to the model
x1 = torch.randn(3, 16, 9, 68)

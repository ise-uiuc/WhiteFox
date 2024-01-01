
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(64, 192, 3, stride=1, padding=1, bias=False)
    def forward(self, x18):
        x1 = self.conv_t(x18)
        x2 = x1 > 0
        x3 = x1 * -6.41359
        x4 = torch.where(x2, x1, x3)
        return torch.nn.functional.relu(x4)
# Inputs to the model
x18 = torch.randn(7, 64, 16, 13)

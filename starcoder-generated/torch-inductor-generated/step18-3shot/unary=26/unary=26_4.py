
class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(7, 7, 3, groups=7, stride=1)
        self.conv = torch.nn.Conv2d(7, 7, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv_t(x)
        x = torch.nn.functional.gelu(x)
        return torch.nn.functional.gelu(self.conv(x))

# Inputs to the model
x = torch.randn(16, 7, 8, 8)


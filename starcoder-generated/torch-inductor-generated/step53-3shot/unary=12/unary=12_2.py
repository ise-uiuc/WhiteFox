
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
        self.conv = torch.nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1)
        self.conv = torch.nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

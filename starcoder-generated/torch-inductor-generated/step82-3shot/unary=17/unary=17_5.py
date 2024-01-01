
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 3, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv1 = torch.nn.ConvTranspose2d(3, 3, 1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.gelu(v1)
        v3 = self.conv1(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 20, 2)

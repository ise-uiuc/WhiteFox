
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(256, 64, 1)
        self.conv2 = torch.nn.ConvTranspose2d(64, 16, 3, stride=2, padding=1, output_padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 256, 1, 1)

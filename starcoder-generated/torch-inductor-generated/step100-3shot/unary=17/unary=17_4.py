
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = torch.nn.functional.interpolate(scale_factor=2)
        self.conv = torch.nn.ConvTranspose2d(3, 128, kernel_size=3, padding=1, stride=1)
    def forward(self, x1):
        v1 = self.upsample(x1)
        v2 = self.conv(v1)
        v3 = torch.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)

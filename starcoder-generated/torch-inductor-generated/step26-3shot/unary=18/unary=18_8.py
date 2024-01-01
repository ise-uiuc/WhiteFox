
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.ConvTranspose2d(
            self.conv1.out_channels,
            self.conv1.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = (self.conv3)(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 128, 128)

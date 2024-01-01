
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(in_channels=7,
            out_channels=8,
            kernel_size=3,
            stride=4,
            padding=(1, 1)
        )
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 7, 31, 33 )

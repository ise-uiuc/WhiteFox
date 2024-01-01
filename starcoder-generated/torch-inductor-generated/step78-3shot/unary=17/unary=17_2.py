
class Model(torch.nn.Module):
    def __init__(self, pad=None, stride=None, is_training=True):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(in_channels=7,
            out_channels=256,
            kernel_size=(1, 2),
            stride=1,
            padding=1,)
        self.conv = torch.nn.Conv2d(in_channels=256,
            out_channels=128,
            kernel_size=(3, 3),
            stride=1,
            padding=3,)
        self.conv2 = torch.nn.ConvTranspose2d(in_channels=128,
            out_channels=7,
            kernel_size=(1, 2),
            stride=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.relu(v1)
        v3 = self.conv(v2)
        v4 = torch.relu(v3)
        v5 = self.conv2(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 7, 257, 257)

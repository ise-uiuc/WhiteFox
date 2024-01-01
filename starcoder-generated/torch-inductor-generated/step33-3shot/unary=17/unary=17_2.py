
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(
            10,
            64,
            kernel_size=4,
            stride=(2, 2),
            padding=(1, 1),
        )
        self.conv2 = torch.nn.ConvTranspose2d(
            64,
            128,
            kernel_size=4,
            stride=(2, 2),
            padding=(1, 1),
            dilation=(2, 2),
        )
        self.conv3 = torch.nn.ConvTranspose2d(
            128,
            64,
            kernel_size=4,
            stride=(2, 2),
            padding=(1, 1),
            dilation=(2, 2),
        )
        self.conv4 = torch.nn.ConvTranspose2d(
            64,
            16,
            kernel_size=4,
            stride=(2, 2),
            padding=(1, 1),
            dilation=(2, 2),
        )
        self.conv5 = torch.nn.ConvTranspose2d(
            16,
            3,
            kernel_size=4,
            stride=(2, 2),
            padding=(1, 1),
            dilation=(2, 2),
        )

    def forward(self, x1):
        x = self.conv1(x1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 10, 64, 64)

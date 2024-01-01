
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            2,
            5,
            (1, 2),
            stride=(1, 2),
            dilation=(1, 2),
            groups=1,
            bias=False,
            padding_mode='zeros',
            padding=(0, 1),
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.conv(x)
        return x
# Inputs to the model
x = torch.randn(1, 2, 5, 7)

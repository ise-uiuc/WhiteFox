
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1,
            dilation=3, groups=1, bias=True
        )
    def forward(self, x):
        x_in = torch.nn.functional.dropout(x, p=0.2, training=True, inplace=False)
        x_out = self.conv(x_in)
        x_out = torch.nn.functional.dropout(x_in, p=0.2, training=True, inplace=False)
        return x_out
# Inputs to the model
x = torch.randn(1, 1, 10, 10)

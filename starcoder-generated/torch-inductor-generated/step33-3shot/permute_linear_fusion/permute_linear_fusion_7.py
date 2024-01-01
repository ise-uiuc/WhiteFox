
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0,), dilation=(1,))
        self.gelu = torch.nn.GELU()
    def forward(self, x1):
        e1 = x1.permute(0, 2, 1)
        e2 = e1.permute(0, 2, 1)
        e2 = e2.permute(0, 2, 3)
        y = self.conv(e2)
        u = 0.712 + y * 0.241 + 0.353 + y
        x = u.flatten(1)
        y = self.gelu(x)
        return y
# Inputs to the model
x1 = torch.randn(1, 1, 2, 2)

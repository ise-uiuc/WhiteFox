
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0,), dilation=(1,))
    def forward(self, x1):
        return self.conv(x1.permute(0, 2, 1))
# Inputs to the model
x1 = torch.randn(1, 2, 2)

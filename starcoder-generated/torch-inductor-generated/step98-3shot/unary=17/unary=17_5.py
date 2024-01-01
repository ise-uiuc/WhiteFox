
class Model(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1):
        super().__init__()
        self.convt = torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
    def forward(self, x):
        v1 = self.convt(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 10, 2, 2)


class Model(torch.nn.Module):
    def __init__(self, channels_in):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 3, 11, padding=5, bias=False)
    def forward(self, x):
        x = self.conv_t(x)
        return x
channels_in = 9
# Input to the model
x = torch.randn(1, channels_in, 12, 12)

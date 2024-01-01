
class ModelTanh(torch.nn.Module):
    def __init__(self, in_channels, conv_dim, out_channels):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose1d(in_channels, out_channels, 3,stride = conv_dim)
    def forward(self, x):
        y = self.conv1(x)
        z = torch.tanh(y)
        return z
# Inputs to the model
x = torch.empty(3, 3, 3).uniform_()

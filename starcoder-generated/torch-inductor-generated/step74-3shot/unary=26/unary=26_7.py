
conv_layers = []
for i in range(36):
    conv_layers.append(
        torch.nn.ConvTranspose2d(32, 32, (3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1))
    )
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = torch.nn.ModuleList(conv_layers)
    def forward(self, x5):
        for layer in self.conv_layers:
            x1 = layer(x5)
            x2 = x1 > 0
            x3 = x1 * 37.94
            x5 = torch.where(x2, x1, x3)
        return x1
# Inputs to the model
x5 = torch.randn(8, 32, 151, 49)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        num_channels = 64
        filters = 3
        num_layers = 6
        self.layers = [torch.nn.ConvTranspose2d(in_channels=num_channels, out_channels=filters, kernel_size=3, stride=(1, 1), padding=(0, 0), dilation=(1, 1), output_padding=(0, 0)) for i in range(num_layers)]
        self.layers = torch.nn.ModuleList(self.layers)
    def forward(self, x):
        out = [x]
        for l in self.layers:
            out.append(l(out[-1]))
        return out[-1]
# Inputs to the model
x1 = torch.randn(1, 60, 224, 224)

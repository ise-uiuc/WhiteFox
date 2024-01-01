
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0, bias=False, dilation=1)
    def forward(self, x):
        x = self.layers(x)
        x = x.view(196, -1)
        x = torch.mean(x, dim=1)
        return x
# Inputs to the model
x = torch.randn(1, 3, 3, 3)

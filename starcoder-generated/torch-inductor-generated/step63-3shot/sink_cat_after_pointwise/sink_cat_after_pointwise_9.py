
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=2, stride=1, padding=0, bias=False)
    def forward(self, x):
        x = self.conv(x)
        y = torch.cat((x, x), dim=1)
        y = y.view(y.shape[0], -1)
        return y.tanh()
# Inputs to the model
x = torch.randn(2, 3, 4, 4)

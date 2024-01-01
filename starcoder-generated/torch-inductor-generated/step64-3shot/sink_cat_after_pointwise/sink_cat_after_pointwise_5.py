
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=2, stride=1, padding=1, bias=True)
    def forward(self, x):
        x_preconcat = self.conv(x)
        x = torch.cat((x, x_preconcat), dim=1)
        return x.relu()
# Inputs to the model
x = torch.randn(1, 3, 4, 4)

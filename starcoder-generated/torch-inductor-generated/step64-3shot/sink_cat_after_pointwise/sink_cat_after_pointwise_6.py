
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0, bias=None)
    def forward(self, x):
        return self.conv(x).transpose(1, 2).view(x.shape[0], 1, -1) if x.shape!= (1, 3, 1, 1) else self.conv(x).transpose(1, 2).view(-1)
# Inputs to the model
x = torch.randn(2, 3, 3, 3)

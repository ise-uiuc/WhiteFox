
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv =  torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=1, padding=1)
    def forward(self, x1):
        v1 = torch.sigmoid(self.conv(x1))
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)

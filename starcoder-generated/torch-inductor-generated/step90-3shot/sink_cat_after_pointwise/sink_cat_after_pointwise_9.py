
class ConcatModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, (1, 10))
    def forward(self, x, y):
        x = self.conv(x)
        y = torch.cat([y.contiguous(), y.contiguous()], dim=1)
        y = y[:, :, :, 0:-4]
        y = self.conv(y)
        return (x, y)
# Inputs to the model
x = torch.randn(10, 1, 224, 224)
y = torch.randn(10, 1, 224, 448)

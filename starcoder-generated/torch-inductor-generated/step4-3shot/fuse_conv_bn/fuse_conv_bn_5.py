
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTransfom(3, 1, 1)
        self.bn = torch.nn.BatchNorm1d(3)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

# Inputs to the model
x = torch.randn(1, 3, 1, 1)

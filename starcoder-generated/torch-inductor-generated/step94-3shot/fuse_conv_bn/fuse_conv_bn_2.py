
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3)
        self.bn = torch.nn.BatchNorm2d(1)
    def forward(self, x5):
        x5 = self.conv(x5)
        x5 = self.bn(x5)
        return nn.functional.dropout(x5, training=True)
# Inputs to the model
x5 = torch.randn(1, 1, 1, 1)

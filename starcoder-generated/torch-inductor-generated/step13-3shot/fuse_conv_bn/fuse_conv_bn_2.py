
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 5, padding=2, padding_mode="replicate")
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x):
        return self.conv(x) * self.bn(x)
# Inputs to the model
x = torch.randn(1, 3, 5, 5)

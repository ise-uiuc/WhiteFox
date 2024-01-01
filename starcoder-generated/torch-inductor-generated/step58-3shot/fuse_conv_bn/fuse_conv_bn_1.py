
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 20, kernel_size=5, padding=2)
        self.bn = torch.nn.BatchNorm2d(20)
    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        return x
# Inputs to the model
input = torch.randn(20, 16, 50, 10)

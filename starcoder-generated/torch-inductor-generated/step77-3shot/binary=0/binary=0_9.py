
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 1, 1, stride=1, padding=1)
    def forward(self, x1, output=[], *args):
        var1 = self.conv(x1)
        output.append(var1)
        return output
# Inputs to the model
x1 = torch.randn(4, 4, 64, 64)

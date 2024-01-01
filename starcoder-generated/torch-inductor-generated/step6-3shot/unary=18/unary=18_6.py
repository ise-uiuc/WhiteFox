
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 1, stride=1, padding=1)
        self.pad = torch.nn.ConstantPad2d(1, -1.0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.pad(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

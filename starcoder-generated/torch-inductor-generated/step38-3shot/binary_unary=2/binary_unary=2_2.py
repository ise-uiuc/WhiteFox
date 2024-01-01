
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(12, 24, 4, stride=4, padding=0)
        self.conv2 = torch.nn.Conv2d(24, 12, 5)
        self.conv2.bias.data = self.conv1.bias.data
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v1 - 0.5
        v4 = v2 - v3
        v5 = F.gelu(v4)
        v6 = v5 - v4
        return v6
# Inputs to the model
x1 = torch.randn(1, 12, 32, 32)

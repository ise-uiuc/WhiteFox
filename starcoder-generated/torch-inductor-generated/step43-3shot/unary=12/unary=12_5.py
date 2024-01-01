
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, (1, 7), stride=1, padding=(0, 3))
        self.conv2 = torch.nn.Conv2d(8, 32, 3, stride=1, padding=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.gelu = torch.nn.GELU()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.sigmoid(v2) * x1
        v4 = self.gelu(v2) + x1
        return v3, v4
# Inputs to the model
x1 = torch.randn(1, 3, 256, 128)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(3, stride=2, padding=1)
        self.conv_pool = torch.nn.Conv2d(32, 64, 1, stride=1, padding=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.pool(v1)
        v3 = self.conv_pool(v2)
        v4 = self.sigmoid(v3)
        v5 = self.tanh(v4)
        return v5 * v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

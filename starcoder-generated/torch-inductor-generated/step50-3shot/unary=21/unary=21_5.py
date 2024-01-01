
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.conv = torch.nn.Conv2d(1, 3, 1, bias=True)
        self.conv2 = torch.nn.Conv2d(1, 8, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 12, 2, stride=2, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.tanh(v1)
        v3 = self.conv2(v2)
        v4 = self.tanh(v3)
        v5 = self.conv3(v4)
        v6 = self.tanh(v5)
        return v6
# Inputs to the model
x = torch.rand(2, 1, 28, 28)


class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.conv1 = torch.nn.Conv2d(64, 192, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(192, 1, 3, stride=3, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v1 = self.tanh(v1)
        v2 = self.conv2(v1)
        v3 = self.sigmoid(v2)
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 64, 14, 14)


class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 5, 2, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(5, 4, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(4, 16, 1, stride=1, padding=1)
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
        self.tanh3 = nn.Tanh()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.tanh1(v3)
        v5 = self.tanh2(v4)
        v6 = self.tanh3(v5)
        return v3
# Inputs to the model
x = torch.randn(64, 8, 10, 10)

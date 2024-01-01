
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(1, 1, 32, stride=2, padding=15)
        self.conv1 = torch.nn.ConvTranspose2d(1, 1, 1, stride=1, padding=1)
        self.max_pool = torch.nn.MaxPool2d(2, 1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv1(v1)
        v3 = torch.tanh(v2)
        v4 = self.max_pool(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)

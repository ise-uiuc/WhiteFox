
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 5, stride=2, padding=1)
        self.act = torch.nn.ReLU6()
        self.conv_next = torch.nn.Conv2d(6, 16, 1, stride=1, padding=1)
    def forward(self, x1):
        x = x1
        x = self.conv(x)
        x = self.act(x)
        x = self.conv_next(x)
        x = F.sigmoid(x)
        x = x * x1
        return x
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

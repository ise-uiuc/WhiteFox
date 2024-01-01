
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
    def forward(self, x):
        x = self.conv(x) + x
        x = torch.sigmoid(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)

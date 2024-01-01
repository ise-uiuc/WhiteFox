
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16,32,3, stride=2, padding=1)
    def forward(self, x):
        x = self.conv(x)
        x = self.conv2(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 128, 128)

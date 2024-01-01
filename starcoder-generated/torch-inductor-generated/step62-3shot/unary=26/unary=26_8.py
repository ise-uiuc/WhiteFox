
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, bias=False)
        self.conv2 = torch.nn.Conv2d(3, 16, 4, stride=2)
        self.conv_t = torch.nn.ConvTranspose2d(16, 16, 4, stride=2)
    def forward(self, x2):
        f1 = self.conv1(x2)
        f2 = self.conv2(f1)
        f3 = self.conv_t(f2)
        return f3
# Inputs to the model
x2 = torch.randn(4, 3, 12, 12)

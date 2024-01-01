
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(128, 16, kernel_size=128, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=128, stride=1, padding=0)
        self.convT = torch.nn.ConvTranspose2d(16, 16, kernel_size=19, stride=1, padding=0)
    def forward(self, x1):
        r1 = self.conv1(x1)
        r2 = self.conv2(r1)
        r3 = self.convT(r2)
        return r3
# Inputs to the model
x1 = torch.randn(1, 128, 28, 28)

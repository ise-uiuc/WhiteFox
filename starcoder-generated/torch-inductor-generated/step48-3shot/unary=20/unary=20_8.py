
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=torch.nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1)
        self.conv_transpose=torch.nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv_transpose(v1)
        v3 = torch.sigmoid(v2)
        return v3
x1 = torch.randn(1, 2, 64, 64)
model=Model()

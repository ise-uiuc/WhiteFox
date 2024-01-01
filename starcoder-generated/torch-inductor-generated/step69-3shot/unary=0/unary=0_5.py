
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(1, 2, 2, stride=2, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        return v1
# Inputs to the model
x = torch.randn(1, 1, 4, 4)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.ConvTranspose2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)

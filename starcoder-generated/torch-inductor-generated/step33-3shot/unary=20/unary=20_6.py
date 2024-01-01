
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = torch.nn.ConvTranspose2d(3, 1, 3, stride=2, padding=1)
        self.deconv2 = torch.nn.ConvTranspose2d(1, 3, 2, stride=2)
    def forward(self, x1):
        v1 = self.deconv1(x1)
        v2 = self.deconv2(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)

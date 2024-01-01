
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.ConvTranspose2d(3, 64, (3, 3))
        self.conv1 = torch.nn.ConvTranspose2d(64, 64, (3, 3))
    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = self.conv1((v1))
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)

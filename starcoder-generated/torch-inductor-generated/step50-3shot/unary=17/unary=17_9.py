
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.ConvTranspose2d(8, 3, 8, padding=4, stride=6)
        self.relu0 = torch.nn.ReLU(inplace=False)
    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = self.relu0(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 8, 10, 10)

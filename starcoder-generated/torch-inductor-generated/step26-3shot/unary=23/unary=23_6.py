
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.ConvTranspose2d(2, 2, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)

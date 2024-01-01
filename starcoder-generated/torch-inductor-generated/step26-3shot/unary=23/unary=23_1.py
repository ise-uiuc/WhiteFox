
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(40, 40, 7, stride=8, padding=0)
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 40, 17, 13)

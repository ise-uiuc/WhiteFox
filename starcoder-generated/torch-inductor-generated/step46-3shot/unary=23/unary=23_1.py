
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.ConvTranspose2d(3, 2, 1, stride=2, padding=1, output_padding=1)
    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 30, 30)

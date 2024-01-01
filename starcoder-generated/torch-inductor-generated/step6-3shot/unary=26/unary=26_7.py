
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(19, 64, 1, stride=1, padding=3)
    def forward(self, x2):
        v1 = self.conv_t(x2)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x2 = torch.randn(1, 19, 8, 8)

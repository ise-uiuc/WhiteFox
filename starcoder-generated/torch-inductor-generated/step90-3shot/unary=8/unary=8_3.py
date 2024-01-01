
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 2, 2, stride=2)
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.tanh(v1)
        v3 = v2 + 3
        v4 = torch.clamp(v3, min=0)
        v5 = torch.clamp(v4, max=6)
        v6 = v5 + v2
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)

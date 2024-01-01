
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose16 = torch.nn.ConvTranspose2d(8, 6, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose16(x1)
        v2 = torch.sigmoid(v1)
        v3 = torch.floor(v1 * v2)
        v4 = v1 - v3
        return torch.tanh(v4)
# Inputs to the model
x1 = torch.randn(1, 8, 5, 3)

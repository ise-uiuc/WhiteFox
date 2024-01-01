
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh2 = torch.nn.Tanh()
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(1, 128, 3, stride=3, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_2(x1)
        v2 = self.tanh2(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)

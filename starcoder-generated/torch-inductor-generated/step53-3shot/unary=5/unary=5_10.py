
# Recurrent neural network
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Initial input
        self.conv_transpose = torch.nn.ConvTranspose2d(12, 100, 3, padding=1)
        # Recurrent layer
        self.conv2d = torch.nn.Conv2d(11, 52, 3, padding=1)
    def forward(self, x1, h1):
        v1 = self.conv_transpose(x1)
        h1 = h1.expand(v1.shape[0], -1, v1.shape[-2], v1.shape[-1])
        v2 = torch.cat([v1, h1], 1)
        v3 = self.conv2d(v2)
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v1 * v5
        return v6, v5
# Inputs to the model
x1 = torch.randn(1, 12, 65, 65)
h1 = torch.randn(1, 100, 2, 2)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(65, 10, 3, stride=1, padding=1)
        self.conv2d = torch.nn.Conv2d(65, 10, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.conv2d(x1)
        v3 = v1 + v2
        v4 = torch.clamp(v3, min=0)
        v5 = torch.clamp(v3, max=6)
        v6 = v4 * v5
        v7 = v6 / 6
        return v7
# Inputs to the model
x1 = torch.randn(1, 65, 90, 89)

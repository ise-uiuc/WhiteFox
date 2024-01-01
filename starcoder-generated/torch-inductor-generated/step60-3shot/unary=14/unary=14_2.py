
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_510_4 = torch.nn.Conv2d(64, 64, 2, stride=2, padding=0)
        self.conv_transpose = torch.nn.ConvTranspose2d(64, 32, 2, stride=2, padding=0, output_padding=1)
    def forward(self, x1):
        v1 = self.conv1_510_4(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 64, 16, 16)

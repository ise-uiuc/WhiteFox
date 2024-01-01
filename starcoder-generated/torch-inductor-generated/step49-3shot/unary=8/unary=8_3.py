
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, 19, stride=15, padding=3, output_padding=8)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.sigmoid(v1)
        v3 = torch.abs(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 36, 62)

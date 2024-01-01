
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 4, 3, padding=1, stride=2, output_padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.conv_transpose(x1)
        v3 = self.conv_transpose(x1)
        v4 = torch.cat((v1, v2, v3), dim=1)
        v5 = torch.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 3, 3)

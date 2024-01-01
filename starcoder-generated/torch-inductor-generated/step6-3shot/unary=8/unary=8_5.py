
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 32, 7, stride=3, padding=1, output_padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.squeeze(v1, dim=2)
        return v2
# Inputs to the model
x1 = torch.randn(1, 8, 24, 24)

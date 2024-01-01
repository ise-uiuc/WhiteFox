
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 8, (3, 5), stride=1, padding=(1, 4), output_padding=(1, 2), bias=True)
    def forward(self, x):
        v1 = self.conv_transpose1(x)
        v2 = self.conv_transpose1(x)
        v4 = v1 * v2
        v5 = v1 + v4
        v8 = v5 * v6
        return v8
# Inputs to the model
x = torch.randn(1, 3, 64, 64)

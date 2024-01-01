
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        return v1 / 5
# Inputs to the model
x1 = torch.randn(1, 128, 64, 64)

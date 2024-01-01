
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        v = self.conv_transpose(x1)
        return v
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

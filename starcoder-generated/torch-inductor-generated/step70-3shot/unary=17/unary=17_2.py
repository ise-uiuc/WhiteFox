
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 8, 3, padding=1, stride=2)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 1000
        return v2
# Inputs to the model
x1 = torch.randn(1, 4, 10, 10)

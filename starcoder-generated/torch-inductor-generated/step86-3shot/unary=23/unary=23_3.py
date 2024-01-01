
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(13, 10, None, stride=(1, 1), padding=(0, 0))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.sin(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 13, 30, 30)

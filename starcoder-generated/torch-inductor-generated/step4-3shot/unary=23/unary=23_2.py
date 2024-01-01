
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 4, 1, stride=1)
        self.conv = torch.nn.Conv2d(4, 5, (3, 5), stride=2)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv(v2)
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 4, 4, 4)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(4, 3, 64, 64)

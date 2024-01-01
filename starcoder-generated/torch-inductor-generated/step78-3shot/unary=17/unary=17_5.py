
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
    def forward(self, x):
        v1 = self.conv_transpose(x)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 64, 64, 64)

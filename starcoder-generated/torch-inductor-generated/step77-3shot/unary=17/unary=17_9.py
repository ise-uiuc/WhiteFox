
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 4, 4, bias=False)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(4, 1, 1, bias=False)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv_transpose1(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 4, 224, 224)

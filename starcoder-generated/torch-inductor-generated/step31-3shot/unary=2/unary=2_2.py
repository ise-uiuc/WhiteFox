
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 4, 2, stride=2)
    def forward(self, x1):
        v1 = F.elu(self.conv_transpose(x1))
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model.
x1 = torch.randn(1, 16, 16, 16)

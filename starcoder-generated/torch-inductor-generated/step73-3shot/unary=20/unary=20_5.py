
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(75, 123, 2, 1, 0)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = F.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 75, 18, 8)

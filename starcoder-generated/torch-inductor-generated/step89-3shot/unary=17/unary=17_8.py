
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 2, 3, stride=1, padding=1, output_padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.transpose(v1, 0, 1)
        v3 = F.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 14, 14)

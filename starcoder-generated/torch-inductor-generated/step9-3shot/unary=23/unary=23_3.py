
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 16, 4, stride=1, padding=2)
        self.tconvtranspose2d = torch.nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1)
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = self.tconvtranspose2d(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)

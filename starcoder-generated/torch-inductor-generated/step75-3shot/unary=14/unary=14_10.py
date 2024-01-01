
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvTranspose1 = torch.nn.ConvTranspose2d(512, 1024, 3, stride=1, padding=0, output_padding=0)
    def forward(self, x1):
        v1 = self.ConvTranspose1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 512, 32, 32)

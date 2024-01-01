
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(28, 4, 5, stride=1, padding=2, output_padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 28, 64, 64)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = torch.relu(x1)
        v2 = self.conv(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 8, 32, 16)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 5, 2, stride=2, padding=0)
    def forward(self, x2):
        v1 = self.conv(x2)
        v2 = v1 - 11.972899436950684
        return v2
# Inputs to the model
x2 = torch.randn(1, 3, 64, 64)

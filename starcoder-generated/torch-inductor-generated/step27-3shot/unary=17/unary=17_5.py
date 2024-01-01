
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 3, 6, stride=1, padding=(1, 2), output_padding=(0, 1))
    def forward(self, x1):
        x2 = self.conv(x1)
        x3 = torch.relu(x2)
        return x3[0, :, 0:20, 25:35]
# Inputs to the model
x1 = torch.randn(1, 3, 40, 40)

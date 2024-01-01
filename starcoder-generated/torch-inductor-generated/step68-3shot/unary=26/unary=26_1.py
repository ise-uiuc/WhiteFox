
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 1, 1, stride=1, padding=0, bias=False)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        z4 = self.conv_t(x1)
        z5 = z4 > 0
        z6 = z4 * -6.75
        z7 = torch.where(z5, z4, z6)
        return self.relu(z7)
# Inputs to the model
x1 = torch.randn(3, 1, 16, 16)

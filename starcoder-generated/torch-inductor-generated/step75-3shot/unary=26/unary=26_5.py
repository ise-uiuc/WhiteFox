
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 48, 55, stride=(3, 2), groups=2, padding=(6, 2))
        self.relu = torch.nn.ReLU()
    def forward(self, x7):
        x1 = self.conv_t(x7)
        x2 = x1 > 0
        x3 = x1 * -4.94
        x4 = torch.where(x2, x1, x3)
        x5 = self.relu(x4)
        return x5
# Inputs to the model
x7 = torch.randn(1, 3, 67, 63)

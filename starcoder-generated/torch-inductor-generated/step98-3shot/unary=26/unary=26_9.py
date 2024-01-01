
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(22, 62, 3, stride=1, padding=1, bias=False)
    def forward(self, x7):
        x1 = self.conv_t(x7)
        x2 = x1 > 0
        x3 = x1 * 0.095
        x4 = torch.where(x2, x1, x3)
        return torch.nn.functional.relu(x4)
# Inputs to the model
x7 = torch.randn(1, 22, 77, 65)

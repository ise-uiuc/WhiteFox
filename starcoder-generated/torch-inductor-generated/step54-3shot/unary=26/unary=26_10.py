
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(7, 1, 8, stride=1, padding=3, bias=False)
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = x1 > 1.67
        x3 = x1 * 1.14
        x4 = torch.where(x2, x1, x3)
        return torch.nn.functional.relu(x4)
# Inputs to the model
x = torch.randn(1, 7, 65, 66)

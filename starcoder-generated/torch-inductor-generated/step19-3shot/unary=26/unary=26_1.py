
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t_1 = torch.nn.ConvTranspose2d(48, 24, 1, stride=2)
        self.conv_t_2 = torch.nn.ConvTranspose2d(24, 12, 1, stride=2)
    def forward(self, x1):
        x2 = self.conv_t_1(x1)
        x3 = self.conv_t_2(x2)
        return x3
# Inputs to the model
x1 = torch.randn(16, 48, 3, 3)

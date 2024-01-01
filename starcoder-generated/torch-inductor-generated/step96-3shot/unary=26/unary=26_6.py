
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(5, 55, 3, stride=1, padding=0, group=1, output_padding=0)
        self.conv_t_relu = torch.nn.ReLU()
    def forward(self, x5):
        b1 = self.conv_t_relu(self.conv_t(x5))
        b2 = b1 > 0
        b3 = b1 * 0.1614
        b4 = torch.where(b2, b1, b3)
        return b4
# Inputs to the model
x5 = torch.randn((3, 5, 59, 39))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(11, 28, 2, bias=False, padding=0)
    def forward(self, input):
        x50 = self.conv_t(input.data)
        x51 = x50 > 0
        x52 = x50 * 0.75
        x53 = torch.where(x51, x50, x52)
        return x53
# Inputs to the model
input = torch.randn(1, 11, 10, 8)

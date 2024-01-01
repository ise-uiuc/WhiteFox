
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(17, 95, 1, padding=0, stride=1, bias=True)
    def forward(self, x51):
        x52 = self.conv_t(x51)
        x53 = x52 > 0
        x54 = x52 * -0.3494
        x55 = torch.where(x53, x52, x54)
        x56 = torch.nn.functional.adaptive_avg_pool2d(x55, 8)
        x57 = self.conv_t(x56)
        x58 = x57 > 0
        x59 = x57 * -0.3014
        x60 = torch.where(x58, x57, x59)
        x61 = torch.reshape(x60, (-1, 249, 484))
        x62 = torch.nn.functional.linear(x61, 13, 1)
        return x62
# Inputs to the model
x51 = torch.randn(8, 17, 9, 9)

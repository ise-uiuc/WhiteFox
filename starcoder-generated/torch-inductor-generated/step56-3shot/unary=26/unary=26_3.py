
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t_1 = torch.nn.ConvTranspose2d(817, 868, 7, stride=1, padding=3, bias=False, dilation=3)
        self.conv_t_2 = torch.nn.ConvTranspose2d(853, 777, 7, stride=3, padding=3, bias=False, dilation=2)
        self.conv_t_3 = torch.nn.ConvTranspose2d(224, 865, 3, stride=1, padding=1, bias=True)
    def forward(self, x10):
        y1 = self.conv_t_1(x10)
        y2 = y1 > 0
        y3 = y1 * -3.277
        y4 = torch.where(y2, y1, y3)
        y5 = torch.cat((y4, torch.nn.functional.adaptive_avg_pool2d(y4, (50, 26))), 1)
        y6 = self.conv_t_2(y5)
        y7 = y6 > 0
        y8 = y6 * 10.437
        y9 = torch.where(y7, y6, y8)
        return self.conv_t_3(y9)
# Inputs to the model
x10 = torch.randn(2, 817, 73, 97)

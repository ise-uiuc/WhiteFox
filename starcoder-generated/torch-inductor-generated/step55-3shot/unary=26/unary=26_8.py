
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(112, 112, 5, stride=2)
        self.conv_t2 = torch.nn.ConvTranspose2d(112, 164, 5, stride=2, groups=4)
        self.conv_t3 = torch.nn.ConvTranspose2d(164, 112, 3, stride=1, padding=1, bias=False)
        self.conv_t4 = torch.nn.ConvTranspose2d(112, 192, 3, stride=1, padding=1, bias=False)
        self.conv_t5 = torch.nn.ConvTranspose2d(192, 112, 3, stride=1, padding=1, bias=False)
        self.conv_t6 = torch.nn.ConvTranspose2d(112, 72, 3, stride=1, padding=1, bias=True)
        self.conv_t7 = torch.nn.ConvTranspose2d(72, 4, 3, stride=1, padding=1, bias=True)
    def forward(self, input_tensor):
        x1 = self.conv_t1(input_tensor)
        x2 = x1 > 0
        x3 = x1 * 0.6995
        x4 = torch.where(x2, x1, x3)
        x5 = self.conv_t2(x4)
        x6 = self.conv_t3(x5)
        x7 = x6 > 0
        x8 = x6 * 1.4029
        x9 = torch.where(x7, x6, x8)
        x10 = self.conv_t4(x9)
        x11 = self.conv_t5(x10)
        x12 = x11 > 0
        x13 = x11 * 1.9844
        x14 = torch.where(x12, x11, x13)
        x15 = self.conv_t6(x14)
        x16 = self.conv_t7(x15)
        return torch.abs(x16)
# Inputs to the model
input_tensor = torch.randn(2, 112, 8, 54)

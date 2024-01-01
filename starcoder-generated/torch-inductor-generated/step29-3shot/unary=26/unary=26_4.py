
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(15, 50, 2, stride=3, padding=0, output_padding=0)
        self.conv_t2 = torch.nn.ConvTranspose2d(50, 35, 2, stride=3, padding=0, output_padding=0)
        self.conv_t3 = torch.nn.ConvTranspose2d(35, 10, 3, stride=3, padding=0, output_padding=0)
        self.conv_t4 = torch.nn.ConvTranspose2d(10, 5, 2, stride=2, padding=1, output_padding=0)
    def forward(self, x1):
        t1 = self.conv_t1(x1)
        t2 = t1 > 0
        t3 = t1 / 0.0396263
        t4 = torch.where(t2, t1, t3)
        t5 = self.conv_t2(t4)
        t6 = t5 > 0
        t7 = t5 / 0.00928939
        t8 = torch.where(t6, t5, t7)
        t9 = self.conv_t3(t8)
        t10 = t9 > 0
        t11 = t9 * 0.0335701
        t12 = torch.where(t10, t9, t11)
        t13 = self.conv_t4(t12)
        return t13
# Inputs to the model
x1 = torch.randn(7, 15, 5, 5)

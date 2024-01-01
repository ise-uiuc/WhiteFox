
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(in_channels=93, out_channels=87, kernel_size=(3, 5), stride=(2, 5), padding=(1, 0))
        self.conv_t1 = torch.nn.ConvTranspose2d(in_channels=87, out_channels=70, kernel_size=(3, 5), stride=(5, 3), padding=(6, 5))
        self.conv_t2 = torch.nn.ConvTranspose2d(in_channels=70, out_channels=93, kernel_size=(3, 5), stride=(4, 1), padding=(1, 7))
    def forward(self, x9):
        t1 = self.conv2(x9)
        t2 = self.conv_t1(t1)
        t3 = torch.nn.functional.gelu(t2)
        t4 = self.conv_t2(t3)
        t5 = torch.sigmoid(t4)
        t6 = torch.tanh(t5)
        t7 = torch.nn.functional.gelu(t2)
        t8 = self.conv_t2(t7)
        t9 = torch.sigmoid(t8)
        t10 = torch.tanh(t9)
        t11 = self.conv_t2(t7)
        t12 = torch.sigmoid(t11)
        t13 = torch.tanh(t12)
        t14 = t10 / t6
        t15 = t13 / t14
        t16 = torch.where(t9 > 0.99438031, t5, t13)
        return t16
# Inputs to the model
x9 = torch.randn(5, 93, 64, 71)

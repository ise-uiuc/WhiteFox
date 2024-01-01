
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(15, 10, (3, 5), stride=(2, 1), padding=(1, 1),
                                                 bias=False, dilation=2, groups=3)
        self.conv_t2 = torch.nn.ConvTranspose2d(10, 5, (4, 3), stride=(1, 2), padding=(3, 1),
                                                 bias=False, dilation=2, groups=2)
        self.conv_t3 = torch.nn.ConvTranspose2d(5, 9, (2, 1), stride=(1, 1), padding=(2, 1),
                                                 bias=False, dilation=1, groups=1)
    def forward(self, input_tensor):
        t1 = self.conv_t1(input_tensor)
        t2 = self.conv_t2(t1)
        t3 = self.conv_t3(t2)
        t4 = t3 > 0
        t5 = t3 * 0.034
        t6 = torch.where(t4, t3, t5)
        t7 = t6 + t1
        return t7
# Inputs to the model
input_tensor = torch.randn(12, 15, 5, 7)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features1 = torch.nn.Conv2d(3, 3, kernel_size=(37, 3), stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.features2 = torch.nn.Conv2d(3, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.features3 = torch.nn.Conv2d(192, 3, kernel_size=(1, 3), stride=(1, 2), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.features4 = torch.nn.Conv2d(3, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.features5 = torch.nn.Conv2d(32, 3, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.features6 = torch.nn.Conv2d(3, 32, kernel_size=(3, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
    def forward(self, v0):
        x1 = self.features1(v0)
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = x3 + x1
        x5 = self.features4(x4)
        x6 = self.features5(x5)
        x7 = x6 + x4
        x8 = self.features6(x7)
        return ((x7), (x8), (x4), (x6), (x2), (x5))
# Inputs to the model
v0 = torch.randn(1, 3, 64, 64)

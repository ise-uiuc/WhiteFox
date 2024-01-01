
class Model(torch.nn.Module):
    def __init__(self, negative_slope, size_average=True, reduce=True):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(63,112,3, stride=1, bias=False, padding=1)
        self.conv_t2 = torch.nn.ConvTranspose2d(112,63,3, stride=2, bias=False, padding=1, dilation=2)
        self.negative_slope = negative_slope
        
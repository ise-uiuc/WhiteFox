
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 2, 4, groups=1, bias=True) 

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(5, 20, 3, stride=1, padding=1, bias=False)
        
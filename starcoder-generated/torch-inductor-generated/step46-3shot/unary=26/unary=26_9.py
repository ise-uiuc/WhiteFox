
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(44, 27, 3, stride=2, padding=0, bias=False)
        self.conv_t_0 = torch.nn.ConvTranspose2d(27, 39, 3, stride=2, padding=0, bias=False)
        self.conv_t_1 = torch.nn.ConvTranspose2d(39, 57, 3, stride=2, padding=0, bias=False)
        self.conv_t_2 = torch.nn.ConvTranspose2d(57, 94, 3, stride=2, padding=0, bias=False)
    def forward(self, x4):
        y0 = torch.nn.Softplus()(self.conv_t(x4))
        y1 = torch.nn.Softplus()(self.conv_t_0(y0))
        y2 = torch.nn.Softplus()(self.conv_t_1(y1))
        return torch.nn.Softplus()(self.conv_t_2(y2))
# Inputs to the model
x4 = torch.randn(7, 44, 399, 745)

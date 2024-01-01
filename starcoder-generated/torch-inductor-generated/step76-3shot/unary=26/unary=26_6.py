
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 6, 1, stride=1, padding=0)
    def forward(self, v):
        u1 = torch.Tensor.transpose(v, 0, 1)
        u2 = u1.contiguous()
        u3 = self.conv_t(u2)
        u4 = u3 > 0
        u5 = u3 * -0.77
        u6 = torch.where(u4, u3, u5)
        return torch.Tensor.transpose(u6, 0, 1)
# Inputs to the model
v = torch.randn(1, 1, 5, 4)

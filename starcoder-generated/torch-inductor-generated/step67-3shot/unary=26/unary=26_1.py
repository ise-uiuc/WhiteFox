
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = nn.ConvTranspose2d(241, 23, 2, stride=1, padding=0, bias=True)
    def forward(self, x17):
        q21 = self.conv_t(x17)
        q22 = q21 > 0
        q23 = q21 * 2.54630
        q24 = torch.where(q22, q21, q23)
        return torch.nn.functional.avg_pool2d(q24, 34)
# Inputs to the model
x17 = torch.randn(54, 241, 36, 31)

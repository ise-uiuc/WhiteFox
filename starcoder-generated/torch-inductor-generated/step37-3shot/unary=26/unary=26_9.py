
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(165, 165, 4, bias=False)
    def forward(self, x15):
        t1 = self.conv_t(x15)
        t2 = t1 > 0
        t3 = t1 * -0.153
        t4 = torch.where(t2, t1, t3)
        return torch.nn.functional.adaptive_avg_pool2d(torch.nn.ReLU()(t4), (1, 1))
# Inputs to the model
x15 = torch.randn(68, 165, 8, 95)

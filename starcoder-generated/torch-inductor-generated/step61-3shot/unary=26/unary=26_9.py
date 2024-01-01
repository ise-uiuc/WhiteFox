
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(128, 25, 2, stride=2)
        self.conv_t2 = torch.nn.ConvTranspose2d(25, 25, 2, stride=2)
    def forward(self, x1):
        t1 = self.conv_t1(x1)
        t2 = t1 > 0
        t3 = t1 * 0.044
        t4 = torch.where(t2, t1, t3)
        t5 = self.conv_t2(t4)
        t6 = t5 > 0
        t7 = t5 * 0.044
        t8 = torch.where(t6, t5, t7)
        return torch.nn.functional.adaptive_avg_pool2d(torch.nn.Softplus()(t8), (1, 1))
# Inputs to the model
x1 = torch.randn(16, 128, 8, 8)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(18, 11, 3, bias=False)
    def forward(self, x21):
        t1 = self.conv_t(x21)
        t2 = t1 > 0
        t3 = t1 * -1284
        t4 = torch.where(t2, t1, t3)
        return torch.nn.functional.avg_pool2d(torch.nn.ReLU()(t4), kernel_size=2)
# Inputs to the model
x21=torch.randn(33, 18, 34, 29)

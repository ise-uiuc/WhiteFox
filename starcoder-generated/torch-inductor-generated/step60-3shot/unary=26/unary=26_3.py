
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(123, 3, 4, stride=1, padding=1, bias=False)
    def forward(self, x):
        t1 = self.conv_t(x)
        t2 = t1 > 0
        t3 = t1 * -0.099
        t4 = torch.where(t2, t1, t3)
        return torch.nn.functional.adaptive_avg_pool3d(torch.nn.ReLU()(t4), (1, 1, 1))
# Inputs to the model
x = torch.randn(2, 123, 22, 248, 281, device='cuda')

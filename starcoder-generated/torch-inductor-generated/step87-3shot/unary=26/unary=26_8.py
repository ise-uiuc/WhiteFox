
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t_1 = torch.nn.ConvTranspose2d(24, 132, 49, stride=2, padding=0, bias=False)
        self.negative_slope = negative_slope
    def forward(self, t1):
        t2 = self.conv_t_1(t1)
        t3 = t2 > 0
        t4 = t2 * self.negative_slope
        t5 = torch.where(t3, t2, t4)
        return torch.flatten(t5, 1)
negative_slope = 0.10000000149011612
# Inputs to the model
t1 = torch.randn(1, 24, 216, 191)

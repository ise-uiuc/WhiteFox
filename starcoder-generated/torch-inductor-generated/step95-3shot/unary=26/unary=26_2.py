
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(38, 112, 10, stride=1, padding=6, bias=False)
        self.linear0 = torch.nn.Linear(13, 34)
    def forward(self, x9):
        m1 = self.conv_t(x9)
        m2 = m1 > 0
        m3 = m1 * 0.29893
        m4 = torch.where(m2, m1, m3)
        m5 = torch.nn.functional.adaptive_avg_pool2d(m4, (4, 4))
        m6 = self.linear0(m5.flatten(1))
        m7 = torch.nn.functional.softmax(m6, dim=-1)

        return m6
# Inputs to the model
x9 = torch.randn(2, 38, 21, 24)

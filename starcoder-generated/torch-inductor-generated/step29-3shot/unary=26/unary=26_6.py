
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(95, 60, 1, stride=1, padding=0)
        self.conv_t2 = torch.nn.ConvTranspose2d(60, 11, 3, stride=2, padding=0)
    def forward(self, x):
        t1 = self.conv_t1(x)
        t2 = torch.clamp(t1, max=3)
        t3 = t1 > 3
        t4 = t1 * 0.5
        t5 = torch.where(t3, t1, t4)
        t6 = t5 == t2
        t7 = t5 > 0
        t8 = t5 * 0.2
        t9 = torch.where(t7, t5, t8)
        x = self.conv_t2(t9)
        return x
# Inputs to the model
x = torch.randn(1, 95, 224, 224)

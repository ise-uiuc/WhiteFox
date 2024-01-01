
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=(3, 2), stride=(3, 1), padding=(2, 3))
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 + 1
        t3 = t2.clamp(0, 6)
        t4 = t1 * t3
        t5 = t4 / 3
        return t5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

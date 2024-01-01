
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=(16, 16), stride=(4, 4), padding=(2, 2))
    def forward(self, x1, x2=None):
        v1 = self.conv(x1)
        t1 = v1.reshape(-1)
        t1 += 1
        other = t1
        if x2!= None:
            v1 = v1 + v2
        else:
            self.v1 = v1
        v2 = self.v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(7, 10, (5, 3), stride=(1, 1), padding=(2, 2))
        self.conv2 = torch.nn.Conv2d(10, 8, 3, stride=(1, 1), padding=(1, 1))
    def forward(self, x):
        negative_slope = 1
        t1 = self.conv1(x)
        t2 = t1 > 0
        t3 = t1 * negative_slope
        f = torch.Tensor([0.46333355])
        v = torch.ones([600])
        z2 = torch.where(t2, t1, v)
        z1 = torch.where(t2, t2, v)
        z3 = torch.where(t2, t3, v)
        z4= z1 + z2
        return self.conv2(z4), torch.where(z4 > f, z4, x)
# Inputs to the model
x1 = torch.randn(1, 7, 10, 5)

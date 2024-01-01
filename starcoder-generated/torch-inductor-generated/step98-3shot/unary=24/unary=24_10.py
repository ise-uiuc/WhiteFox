
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        conv = torch.nn.Conv2d(1, 2, (3, 1), stride=1, padding=2, bias=False)
        t1 = conv(x)
        t2 = t1 > 0
        t3 = t1 * - 0.2629397
        t4 = torch.where(t2, t1, t3)
        return t4
# Inputs to the model
x1 = torch.randn(6, 1, 13, 198)

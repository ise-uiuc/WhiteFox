
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.gelu = torch.nn.GELU()
    def forward(self, x1):
        t1 = self.conv2d(x1)
        t2 = self.gelu(t1)
        s1 = torch.sigmoid(t2)
        m1 = torch.mul(s1, t2)
        return m1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

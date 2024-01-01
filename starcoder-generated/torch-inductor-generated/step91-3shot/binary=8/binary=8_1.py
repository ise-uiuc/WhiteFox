
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        l1 = self.conv1(x1)
        m1 = self.conv2(x2)
        h1 = torch.add(l1, m1)
        l2 = self.conv1(x1.sub(torch.add(x1, m1)))
        m2 = self.conv2(x2.mul(x1))
        h2 = torch.add(l2, m2)
        l3 = self.conv1(x2)
        m3 = self.conv2(x1)
        h3 = torch.add(l3, m3)
        l4 = self.conv1(x2.sub(torch.add(x2, m3)))
        m4 = self.conv2(x1.mul(x2))
        h4 = torch.add(l4, m4)
        h5 = h1 + h2
        h6 = h3 + h4
        return h2 + h5
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)

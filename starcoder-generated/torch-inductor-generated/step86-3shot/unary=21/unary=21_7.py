
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, )
        self.conv2 = torch.nn.Conv2d(3, 3, (1, 2))
    def forward(self, x):
        n1 = self.conv(x)
        n2 = torch.tanh(n1)
        n3 = self.conv2(n1)
        n4 = torch.tanh(n1)
        n5 = torch.cat([n2, n3, n4], 1)
        return n5
# Inputs to the model
x = torch.randn(1, 3, 64, 64)

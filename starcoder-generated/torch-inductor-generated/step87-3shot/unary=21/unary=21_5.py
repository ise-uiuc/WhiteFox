
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convA = torch.nn.Conv2d(8, 8, 3, stride=2, padding=1)
        self.convB = torch.nn.Conv2d(8, 8, 3, stride=2, padding=1)
    def forward(self, x):
        a = self.convA(x)
        b1 = torch.tanh(a)
        b2 = torch.tanh(b1)
        c = self.convB(b2)
        d1 = torch.tanh(c)
        d2 = torch.tanh(d1)
        return d2
# Inputs to the model
x = torch.randn(1, 8, 32, 32)

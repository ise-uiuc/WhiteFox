
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.a = torch.nn.BatchNorm2d(2, momentum=0)
        torch.manual_seed(1)
        self.conv = torch.nn.Conv2d(1, 1, 7, bias=Fals)
        torch.manual_seed(1)
        self.b = torch.nn.BatchNorm2d(1)
    def forward(self, x):
        x = self.a(x)
        x = self.conv(x)
        a = self.b(x)
        b = self.b(x)
        c = self.a(x, a, b)
        a = self.b(x)
        return a, a
# Inputs to the model
x = torch.randn(2, 1, 10, 20)

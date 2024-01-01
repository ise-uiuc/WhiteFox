
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 7, stride=1, padding=3, groups=2)
        self.tanh = torch.nn.Tanh()
    def forward(self, x14):
        t1 = self.conv(x14)
        t2 = self.tanh(t1)
        return t2
# Inputs to the model
x14 = torch.randn(1, 3, 49, 89)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.act1 = torch.nn.Sigmoid()
        self.act2 = torch.nn.Hardtanh()
    def forward(self, x1):
        t1 = self.conv1(x1)
        t2 = self.act1(t1)
        t3 = self.act2(t1)
        t4 = t2 + t3
        return t4
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(113)
        self.layer1 = torch.nn.Conv2d(6, 6, 1, bias=True)
        torch.manual_seed(11)
        self.layer2 = torch.nn.BatchNorm2d(6)
    def forward(self, x2):
        s2 = self.layer(x2)
        s2 = self.layer(s2)
        x2 = s2 + s2
# Inputs to the model
x2 = torch.randn(1, 6, 6, 6)

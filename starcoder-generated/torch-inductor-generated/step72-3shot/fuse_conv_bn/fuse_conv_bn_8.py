
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(3)
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, 3),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU(inplace=True))
    def forward(self, x3):
        o = self.layer(x3)
        return o*o*o + o + 0.6
# Inputs to the model
x3 = torch.randn(1, 3, 4, 4)

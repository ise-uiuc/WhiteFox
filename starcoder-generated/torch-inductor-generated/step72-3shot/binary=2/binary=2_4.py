
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 80, 7, stride=9, padding=0)
    def forward(self, x):
        v1 = self.conv(x)
        return v1 - -36.475
# Inputs to the model
x = torch.randn(1, 3, 19, 80)

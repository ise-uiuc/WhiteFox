
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, (1,5), stride=1, padding=(0,2))
        self.conv2 = torch.nn.Conv2d(64, 64, (5,1), stride=1, padding=(2,0))
    def forward(self, x0):
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = x2 + 3
        x4 = torch.clamp_min(x3, 0)
        x5 = torch.clamp_max(x4, 6)
        x6 = x2 * x5
        x7 = x6 / 6
        return x7
# Inputs to the model
x0 = torch.randn(1, 3, 64, 64)

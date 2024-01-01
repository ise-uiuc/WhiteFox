
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, (3,3), stride=1, padding=(1,1))
    def forward(self, x):
        v = self.conv(x)
        return v - 3.3959
# Inputs to the model
x = torch.randn(1, 3, 64, 64)

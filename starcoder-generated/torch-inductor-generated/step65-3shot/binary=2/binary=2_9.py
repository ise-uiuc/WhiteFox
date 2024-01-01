
class Model(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 6, stride=8, padding=10)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 9.1
        return v2
# Inputs to the model
x = torch.randn(2, 3, 20, 20)

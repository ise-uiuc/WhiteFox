
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, stride=2)
    def forward(self, x1):
        s1 = self.conv(x1 + x1)
        return s1 
# Inputs to the model
x1 = torch.randn(1, 1, 4, 4)

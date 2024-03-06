
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 1, 2, stride=3)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1
        return v2    
# Inputs to the model
x1 = torch.randn(1, 2, 256, 256)
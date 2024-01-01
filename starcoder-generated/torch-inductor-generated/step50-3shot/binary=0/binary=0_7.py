
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 16, 2, stride=2, padding=1)
    def forward(self, x1, other=1):
        v1 = self.conv(x1)
    
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)

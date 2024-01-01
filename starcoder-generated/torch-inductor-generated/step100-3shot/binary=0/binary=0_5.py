
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 13, 7, stride=1, padding=3)
    def forward(self, x=torch.randn(1, 1, 124, 124)):
        return self.conv(x)
# Inputs to the model

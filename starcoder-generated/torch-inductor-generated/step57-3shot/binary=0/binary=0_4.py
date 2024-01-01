
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d( 7, 12, 1, stride=2, padding=1 )
    def forward(self, x1, padding=None):
        v1 = self.conv(x1)
        v2 = v1.squeeze(2)
        return v2
# Inputs to the model
x1 = torch.randn(1, 7, 1, 1)

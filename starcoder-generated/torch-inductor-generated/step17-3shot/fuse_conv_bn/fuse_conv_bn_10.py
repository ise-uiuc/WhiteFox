
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(6, 2, kernel_size=3, stride=2), nn.BatchNorm2d(6),)
    def forward(self, x):
        return self.layer(x)
# Inputs to the model
x = torch.randn(1, 6, 4, 4)

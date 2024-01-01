
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, 1, 2, 2, 0))
        layers.append(nn.BatchNorm2d(3, eps=0.004, momentum=0, affine=True, track_running_stats=True))
        self.conv1 = nn.Sequential(*layers[0:1])
    def forward(self, x):
        return self.conv1(x)
# Inputs to the model
x = torch.randn(1, 3, 2, 2)

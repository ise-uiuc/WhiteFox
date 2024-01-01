
class BNOptimizeModel(nn.Module):
    def __init__(self):
        super(BNOptimizeModel, self).__init__()
        self.bn = nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    def forward(self, x):
        x = torch.add(x, 1)
        x = self.bn(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 4, 4)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(torch.nn.Conv3d(3, 3, kernel_size=3, stride=1, padding=1), torch.nn.BatchNorm3d(3, eps=1e-05, momentum=0.1, affine=True))
        self.conv2 = nn.Sequential(*[nn.Linear(3, 3, bias=False), nn.BatchNorm1d(3)])
    def forward(self, x):
        return self.conv2(self.conv1(x).sum([2, 3]))
# Inputs to the model
x = torch.randn(1, 3, 4, 4, 4)

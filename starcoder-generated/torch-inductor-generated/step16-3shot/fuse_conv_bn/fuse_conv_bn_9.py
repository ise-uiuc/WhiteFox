
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = nn.Sequential(nn.Conv3d(3, 3, kernel_size=3, stride=1, padding=1), nn.BatchNorm3d(3, eps=1e-05, momentum=0.1, affine=True), nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True), nn.ReLU())
    def forward(self, x):
        return self.layer(x)
# Inputs to the model
x = torch.randn(1, 3, 4, 4, 4)

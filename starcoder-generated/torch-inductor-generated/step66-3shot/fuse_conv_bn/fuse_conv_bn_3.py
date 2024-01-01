
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 128, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(128, eps=1e-5, momentum=0.10000000000000001, affine=True, track_running_stats=True), nn.Conv2d(128, 512, kernel_size=3, stride=2, padding=1, bias=False))
        self.classifier = nn.Sequential(nn.Linear(1280, 512, bias=True))
    def forward(self, x):
        x = self.features(x)
        x = nn.Flatten()(x)
        x = self.classifier(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 32, 32)

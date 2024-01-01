
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        conv = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1, bias=False)
        bn = nn.BatchNorm2d(3)
        self.layer = nn.Sequential(conv, bn)
    def forward(self, x):
        return self.layer(x)
# Inputs to the model
x = torch.randn(1, 3, 4, 4)

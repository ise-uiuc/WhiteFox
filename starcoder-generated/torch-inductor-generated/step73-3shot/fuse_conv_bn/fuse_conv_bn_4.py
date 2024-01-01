
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1)
        self.norm = nn.BatchNorm2d(num_features=3)
    def forward(self, x):
        x = self.conv(x)
        a = torch.flatten(x, 1, -1)
        return self.norm(a)
# Inputs to the model
x = torch.randn(1, 3, 4, 4)

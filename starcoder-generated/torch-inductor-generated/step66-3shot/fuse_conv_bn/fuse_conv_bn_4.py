
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1, )
        self.norm = nn.BatchNorm2d(num_features=1, )
    def forward(self, x):
        # x.shape = [1, 3, 4, 4]
        a = self.conv(x)
        # a.shape = [1, 1, 4, 4]
        b = self.norm(a)
        # b.shape = [1, 1, 4, 4]
        return b
# Inputs to the model
x = torch.randn(1, 3, 4, 4)

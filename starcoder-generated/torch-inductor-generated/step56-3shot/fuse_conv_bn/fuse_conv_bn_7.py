
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(54, 44, kernel_size=(1, 1))
        self.bn1 = nn.BatchNorm2d(44)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        return self.relu(out)

model = Model()
# Inputs to the model
input_ = torch.randn(8, 54, 34, 14)

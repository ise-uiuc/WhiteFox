
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1  = nn.ConvTranspose1d(1, 64, 2, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=False)
        self.relu  = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 10)

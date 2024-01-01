
class ModelSigmoid(torch.nn.Module):
    def __init__(self):
        super(ModelSigmoid, self).__init__()
        self.conv = nn.Conv2d(2, 5, kernel_size=2, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(5)
        self.pool = nn.MaxPool2d(2)
        self.deconv = nn.ConvTranspose2d(5, 2, kernel_size=2, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = F.relu(v2)
        v4 = self.pool(v3)
        v5 = F.tanh(v4)
        v6 = self.deconv(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 2, 11, 11)

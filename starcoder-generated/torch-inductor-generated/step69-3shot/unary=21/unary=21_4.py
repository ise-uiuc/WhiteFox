
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, input):
        feature = self.conv1(input)
        feature = self.relu(feature)
        return feature
model = Model()
# Inputs to the model
x = torch.randn(2, 32, 100, 100)

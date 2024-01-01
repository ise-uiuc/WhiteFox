
class ModelTanh(nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv1 = nn.Conv2d(3,64,3)
        self.conv2 = nn.Conv2d(64,64,3)
        self.conv3 = nn.Conv2d(64,3,3)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.conv2(x)
        x = self.tanh(x)
        x = self.conv3(x)
        x = self.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 224, 224)

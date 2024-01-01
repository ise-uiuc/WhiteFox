
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 30, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(30, 54, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(54, 30, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(30, 10, 1, stride=1, padding=0)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        v1 = self.conv1(x)
        v1 = self.tanh(v1)
        v2 = self.conv2(v1)
        v2 = self.tanh(v2)
        v3 = self.conv3(v2)
        v3 = self.tanh(v3)
        v4 = self.conv4(v3)
        v4 = self.tanh(v4)
        return v4
# Inputs to the model
x = torch.randn(1, 3, 60, 60)

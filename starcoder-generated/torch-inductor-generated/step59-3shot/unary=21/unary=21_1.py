
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = torch.nn.ReflectionPad2d(3)
        self.conv1 = torch.nn.Conv2d(1, 3, 2, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 3, 2, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(3, 5, 3, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(5, 5, 3, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(5, 7, 5, stride=1, padding=0)
    def forward(self, x):
        v1 = self.pad(x)
        v2 = self.conv1(v1)
        v3 = torch.tanh(v2)
        v4 = self.conv2(v3)
        v5 = torch.tanh(v4)
        v6 = self.conv3(v5)
        v7 = torch.tanh(v6)
        v8 = self.conv4(v7)
        v9 = torch.tanh(v8)
        v10 = self.conv5(v9)
        return v10
# Inputs to the model
x = torch.randn(1, 1, 10, 10)

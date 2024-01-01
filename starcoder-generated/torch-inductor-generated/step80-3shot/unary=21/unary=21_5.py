
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 256, 1, padding=(0, 1), stride=1)
        self.conv2 = torch.nn.Conv2d(256, 256, (1, 5), padding=(0, 2), stride=1)
        self.conv3 = torch.nn.Conv2d(256, 3, (2, 2), padding=(0, 3), stride=1)
        self.conv4 = torch.nn.Conv2d(3, 256, 1, padding=(0, 1), stride=1)
        self.conv5 = torch.nn.Conv2d(256, 256, (2, 4), padding=(0, 4), stride=1)
        self.conv6 = torch.nn.Conv2d(256, 3, (1, 1), padding=(0, 1), stride=2)
    def forward(self, x1):
        v2 = self.conv1(x1)
        v2 = torch.tanh(v2)
        v2 = self.conv2(v2)
        v2 = torch.tanh(v2)
        v2 = self.conv3(v2)
        v3 = self.conv4(x1)
        v3 = torch.tanh(v3)
        v3 = self.conv5(v3)
        v3 = torch.tanh(v3)
        v3 = self.conv6(v3)
        v4 = torch.tanh(v2 + v3)
        return v2
# Inputs to the model
x = torch.randn(10, 3, 224, 224)

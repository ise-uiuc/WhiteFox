
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 8, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 4, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = v3 - 3.3
        v5 = F.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)

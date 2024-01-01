
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(512, 256, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(256, 512, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = F.relu(self.conv1(x))
        v2 = F.relu(self.conv2(v1))
        v3 = v2 - 0.5
        return v3
# Inputs to the model
x1 = torch.randn(1, 512, 7, 7)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 4, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(4, 4, 5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(4, 4, 5, stride=1, padding=2)
        self.conv4 = torch.nn.Conv2d(4, 4, 5, stride=1, padding=2)
    def forward(self, x1):
        v1 = torch.relu(self.conv1(x1))
        v2 = torch.relu(self.conv2(v1))
        v3 = torch.relu(self.conv3(v2))
        v4 = torch.relu(self.conv4(v3))
        v5 = self.conv4(x1)
        return v5
# Inputs to the model
x1 = torch.randn(1, 4, 128, 128)

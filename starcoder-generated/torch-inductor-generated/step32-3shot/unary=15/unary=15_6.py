
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 35, 5, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(35, 35, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(35, 6, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv3(v2)
        v4 = torch.nn.functional.softmax(v3, dim=1)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)

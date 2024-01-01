
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(3, 6, 3, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(6, 8, 3, stride=1, padding=1)
        self.conv2_3 = torch.nn.Conv2d(6, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        x1 = self.conv2(x1)
        v1 = self.conv1(x1)
        x1 = self.conv2_3(x1)
        v1 = v1 - x1
        v3 = F.relu(v1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 30, 30)
